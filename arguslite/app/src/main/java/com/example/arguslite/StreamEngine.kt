package com.example.arguslite

import android.graphics.Bitmap
import kotlin.math.sqrt

/**
 * Streaming version of the study pipeline: frames arrive one at a time (from a video
 * being stepped through, or a live camera) and the map, coverage counts, and findings
 * grow with each frame. Same math and same frozen gates as StudyEngine — this class
 * only changes WHEN work happens, not what the work is. Coverage is counted on every
 * processed frame, so the 80-frame sampling cap of batch studies does not apply here.
 *
 * The static-pixel mask (overlay/vignette) is estimated online with a running
 * mean/variance instead of a whole-clip pass; it becomes active after a short warm-up.
 */
class StreamEngine(depthEngine: DepthEngine, private val polyp: PolypEngine) {

    private val study = StudyEngine(depthEngine, polyp)
    private val depthNet = depthEngine
    private val W = study.W
    private val F = study.F

    private class Det(val cx: Float, val cy: Float, val cz: Float,
                      val wx: Float, val wy: Float, val wz: Float,
                      val sizeMm: Float, val conf: Float, val frame: Int)

    // ---- accumulating state (all guarded by synchronized(this)) ----
    private var n = 0            // processed frames (after warm-up)
    private var warm = 0         // calibration-only frames
    private val WARMUP = 12
    private var prevGray: FloatArray? = null
    private var prevDepth: FloatArray? = null
    private var pose = StudyEngine.Pose.identity()
    private var scale = -1f
    private var pathMm = 0f
    private val views = HashMap<Long, Int>()
    private val colorMap = HashMap<Long, Int>()
    private val bornMap = HashMap<Long, Int>()
    private val tsdf = HashMap<Long, Long>()
    private var meshCache: MeshData? = null
    private var meshBuiltAt = -1
    private val dets = ArrayList<Det>()
    private var detFrames = 0
    private val trajList = ArrayList<Float>()
    private val statVoxels = ArrayList<Int>()
    private val statBlind = ArrayList<Float>()
    private val statPath = ArrayList<Float>()
    private val thumbs = ArrayList<Bitmap>()
    private var thumbStride = 1
    private val runMean = FloatArray(W * W)
    private val runM2 = FloatArray(W * W)
    private val staticMask = BooleanArray(W * W)
    private val stageSecs = FloatArray(4)
    var lastFrame: Bitmap? = null
        private set
    @Volatile var busy = false
        private set

    private val DETECT_EVERY = 3

    fun step(bmp: Bitmap) {
        synchronized(this) {
            busy = true
            try {
                stepLocked(bmp)
            } finally {
                busy = false
            }
        }
    }

    private fun stepLocked(bmp: Bitmap) {
        var t0 = System.nanoTime()
        val small = Bitmap.createScaledBitmap(bmp, W, W, true)
        val px = IntArray(W * W)
        small.getPixels(px, 0, W, 0, 0, W, W)
        val gray = FloatArray(W * W)
        for (j in px.indices) {
            val r = (px[j] shr 16) and 0xFF
            val g = (px[j] shr 8) and 0xFF
            val b = px[j] and 0xFF
            gray[j] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
        }
        // online static mask (Welford), continuously refreshed
        val seenTotal = warm + n + 1
        for (j in gray.indices) {
            val d = gray[j] - runMean[j]
            runMean[j] += d / seenTotal
            runM2[j] += d * (gray[j] - runMean[j])
        }
        // Warm-up: the first frames only calibrate the static mask, so the depth scale
        // (and everything downstream) is set from clean scene pixels — the same
        // mask-first-scale-second ordering the batch pipeline uses.
        if (warm < WARMUP) {
            warm++
            lastFrame = bmp
            if (warm == WARMUP) {
                for (j in gray.indices) staticMask[j] = runM2[j] / seenTotal < 2e-4f
            }
            return
        }
        for (j in gray.indices) staticMask[j] = runM2[j] / seenTotal < 2e-4f
        val disp = depthNet.disparityGrid(bmp, W)
        if (scale < 0f) {
            val inv = ArrayList<Float>(W * W)
            for (j in disp.indices)
                if (!staticMask[j] && disp[j] > 1e-6f && gray[j] < study.GLARE)
                    inv.add(1f / disp[j])
            inv.sort()
            scale = 35f / (if (inv.isEmpty()) 1f else inv[inv.size / 2])
        }
        val depth = FloatArray(W * W)
        for (j in disp.indices) depth[j] = if (disp[j] > 1e-6f) scale / disp[j] else 0f
        stageSecs[0] += (System.nanoTime() - t0) / 1e9f

        // tracking
        t0 = System.nanoTime()
        val pg = prevGray; val pd = prevDepth
        if (pg != null && pd != null) {
            val rel = Vo.trackPair(pg, pd, gray, staticMask)
            val next = pose.compose(rel.inverse())
            val dx = next.t[0] - pose.t[0]; val dy = next.t[1] - pose.t[1]
            val dz = next.t[2] - pose.t[2]
            pathMm += sqrt(dx * dx + dy * dy + dz * dz)
            pose = next
        }
        prevGray = gray; prevDepth = depth
        trajList.add(pose.t[0]); trajList.add(pose.t[1]); trajList.add(pose.t[2])
        stageSecs[1] += (System.nanoTime() - t0) / 1e9f

        // fusion + view counting (every frame — coverage without sampling caps)
        t0 = System.nanoTime()
        val seen = HashSet<Long>()
        var y = 0
        while (y < W) {
            var x = 0
            while (x < W) {
                val j = y * W + x
                val z = depth[j]
                if (!staticMask[j] && z > 5f && z < 140f && gray[j] < study.GLARE) {
                    val lx = (x - study.CX) / F * z
                    val ly = (y - study.CY) / F * z
                    val wx = pose.r[0] * lx + pose.r[1] * ly + pose.r[2] * z + pose.t[0]
                    val wy = pose.r[3] * lx + pose.r[4] * ly + pose.r[5] * z + pose.t[1]
                    val wz = pose.r[6] * lx + pose.r[7] * ly + pose.r[8] * z + pose.t[2]
                    val key = study.voxelKey(wx, wy, wz)
                    if (seen.add(key)) {
                        views[key] = (views[key] ?: 0) + 1
                        if (!bornMap.containsKey(key)) bornMap[key] = n
                    }
                    colorMap[key] = px[j]
                    // signed-distance band along the viewing ray, for the surface mesh
                    var rx = wx - pose.t[0]; var ry = wy - pose.t[1]; var rz = wz - pose.t[2]
                    val rl = sqrt(rx * rx + ry * ry + rz * rz)
                    if (rl > 1e-3f) {
                        rx /= rl; ry /= rl; rz /= rl
                        val trunc = TSDF_TRUNC_VOXELS * study.VOXEL
                        for (mstep in -2..2) {
                            val h = mstep * study.VOXEL
                            val kk = study.voxelKey(wx - rx * h, wy - ry * h, wz - rz * h)
                            val sd = (h / trunc).coerceIn(-1f, 1f)
                            val prev = tsdf[kk]
                            tsdf[kk] = if (prev == null) tsdfPack(sd, 1f)
                            else tsdfPack(tsdfSum(prev) + sd, tsdfWeight(prev) + 1f)
                        }
                    }
                }
                x += 2
            }
            y += 2
        }
        if (n - meshBuiltAt >= 12) {
            meshCache = buildSurface(tsdf, colorMap, views, study::voxelCenter)
            meshBuiltAt = n
        }
        statVoxels.add(views.size)
        var bl = 0
        for (v in views.values) if (v < study.BLIND_VIEWS) bl++
        statBlind.add(100f * bl / maxOf(1, views.size))
        statPath.add(pathMm)
        stageSecs[2] += (System.nanoTime() - t0) / 1e9f

        // detection at the frozen gates, every few frames
        if (n % DETECT_EVERY == 0) {
            t0 = System.nanoTime()
            val r = polyp.run(bmp, staticMask, W)
            if (r.found && r.avgConf >= 0.7f) {
                val gx = (r.cx * W).toInt().coerceIn(0, W - 1)
                val gy = (r.cy * W).toInt().coerceIn(0, W - 1)
                val zs = ArrayList<Float>()
                r.maskGrid?.let { mg ->
                    for (j in mg.indices) if (mg[j] && depth[j] > 0f) zs.add(depth[j])
                }
                val z = if (zs.isNotEmpty()) { zs.sort(); zs[zs.size / 2] }
                else depth[gy * W + gx]
                if (z > 5f && z < 140f) {
                    val boxNorm = maxOf(r.boxW.toFloat() / r.outW, r.boxH.toFloat() / r.outH)
                    val sizeMm = boxNorm * W / F * z
                    if (sizeMm in 2f..60f) {
                        detFrames++
                        val lx = (gx - study.CX) / F * z
                        val ly = (gy - study.CY) / F * z
                        dets.add(Det(lx, ly, z,
                            pose.r[0]*lx + pose.r[1]*ly + pose.r[2]*z + pose.t[0],
                            pose.r[3]*lx + pose.r[4]*ly + pose.r[5]*z + pose.t[1],
                            pose.r[6]*lx + pose.r[7]*ly + pose.r[8]*z + pose.t[2],
                            sizeMm, r.avgConf, n))
                    }
                }
            }
            stageSecs[3] += (System.nanoTime() - t0) / 1e9f
        }

        // thumbnails with stride doubling so memory stays bounded
        if (n % thumbStride == 0) {
            thumbs.add(Bitmap.createScaledBitmap(bmp, 224,
                maxOf(1, 224 * bmp.height / bmp.width), true))
            if (thumbs.size >= 120) {
                val kept = ArrayList<Bitmap>(60)
                for (i in thumbs.indices step 2) kept.add(thumbs[i])
                thumbs.clear(); thumbs.addAll(kept)
                thumbStride *= 2
            }
        }
        lastFrame = bmp
        n++
    }

    // ---- lesion clustering at the frozen rule (camera-frame ball, scaled support) ----
    private var bestCluster = 0
    private fun lesions(): List<StudyEngine.Lesion> {
        val minSupport = maxOf(6, n / 5)
        val used = BooleanArray(dets.size)
        val out = ArrayList<StudyEngine.Lesion>()
        bestCluster = 0
        for (i in dets.indices) {
            if (used[i]) continue
            val grp = ArrayList<Det>()
            for (j in i until dets.size) {
                if (used[j]) continue
                val dx = dets[j].cx - dets[i].cx; val dy = dets[j].cy - dets[i].cy
                val dz = dets[j].cz - dets[i].cz
                if (dx*dx + dy*dy + dz*dz < 100f) { grp.add(dets[j]); used[j] = true }
            }
            if (grp.size > bestCluster) bestCluster = grp.size
            if (grp.size >= minSupport) {
                val sizes = grp.map { it.sizeMm }.sorted()
                val frames0 = grp.map { it.frame }.sorted()
                out.add(StudyEngine.Lesion(
                    grp.map { it.wx }.average().toFloat(),
                    grp.map { it.wy }.average().toFloat(),
                    grp.map { it.wz }.average().toFloat(),
                    sizes[sizes.size / 2], grp.size,
                    grp.map { it.conf }.average().toFloat(), frames0[frames0.size / 2]))
            }
        }
        return out
    }

    class LiveState(val frames: Int, val voxels: Int, val blindPct: Float,
                    val pathMm: Float, val pts: FloatArray, val colors: IntArray,
                    val blind: BooleanArray, val traj: FloatArray,
                    val lesions: List<StudyEngine.Lesion>, val stageMs: FloatArray,
                    val thumb: Bitmap?, val warming: Boolean, val warmDone: Int,
                    val warmTotal: Int, val bestSupport: Int, val needSupport: Int,
                    val viewsArr: IntArray, val spanMm: Float,
                    val mesh: MeshData?, val cX: Float, val cY: Float, val cZ: Float)

    fun snapshot(): LiveState = synchronized(this) {
        val keys = ArrayList(views.keys)
        val stride = maxOf(1, keys.size / 6000)
        val sel = ArrayList<Long>((keys.size / stride) + 1)
        for (i in keys.indices step stride) sel.add(keys[i])
        var mnx = Float.MAX_VALUE; var mny = Float.MAX_VALUE; var mnz = Float.MAX_VALUE
        var mxx = -Float.MAX_VALUE; var mxy = -Float.MAX_VALUE; var mxz = -Float.MAX_VALUE
        val raw = FloatArray(3 * sel.size)
        for ((i, key) in sel.withIndex()) {
            val (vx, vy, vz) = study.voxelCenter(key)
            raw[3*i] = vx; raw[3*i+1] = vy; raw[3*i+2] = vz
            if (vx < mnx) mnx = vx; if (vx > mxx) mxx = vx
            if (vy < mny) mny = vy; if (vy > mxy) mxy = vy
            if (vz < mnz) mnz = vz; if (vz > mxz) mxz = vz
        }
        val cx = (mnx + mxx) / 2f; val cy = (mny + mxy) / 2f; val cz = (mnz + mxz) / 2f
        val span = maxOf(mxx - mnx, mxy - mny, mxz - mnz, 1e-3f) / 2f
        val pts = FloatArray(3 * sel.size)
        val cols = IntArray(sel.size)
        val blind = BooleanArray(sel.size)
        val vcs = IntArray(sel.size)
        for (i in sel.indices) {
            pts[3*i] = (raw[3*i] - cx) / span
            pts[3*i+1] = (raw[3*i+1] - cy) / span
            pts[3*i+2] = (raw[3*i+2] - cz) / span
            cols[i] = colorMap[sel[i]] ?: 0
            vcs[i] = views[sel[i]] ?: 0
            blind[i] = vcs[i] < study.BLIND_VIEWS
        }
        var blAll = 0
        for (v in views.values) if (v < study.BLIND_VIEWS) blAll++
        val traj = FloatArray(trajList.size)
        for (i in traj.indices step 3) {
            traj[i] = (trajList[i] - cx) / span
            traj[i+1] = (trajList[i+1] - cy) / span
            traj[i+2] = (trajList[i+2] - cz) / span
        }
        val les = lesions().map { StudyEngine.Lesion(
            (it.x - cx) / span, (it.y - cy) / span, (it.z - cz) / span,
            it.sizeMm, it.support, it.avgConf, it.frame) }
        val ms = FloatArray(4)
        for (i in 0..3) ms[i] = 1000f * stageSecs[i] / maxOf(1, n)
        LiveState(n, views.size, 100f * blAll / maxOf(1, views.size), pathMm,
            pts, cols, blind, traj, les, ms, lastFrame,
            warm < WARMUP, warm, WARMUP, bestCluster, maxOf(6, n / 5), vcs, span,
            meshCache, cx, cy, cz)
    }

    /** Convert the accumulated stream into a StudyResult so the Studies tab can replay it. */
    fun finish(totalSecs: Float): StudyEngine.StudyResult = synchronized(this) {
        meshCache = buildSurface(tsdf, colorMap, views, study::voxelCenter)
        meshBuiltAt = n
        val s = snapshot()
        // compress the timeline to the thumbnail timeline
        val tn = maxOf(1, thumbs.size)
        fun mapFrame(f: Int) = (f / thumbStride).coerceIn(0, tn - 1)
        val born = IntArray(s.pts.size / 3)
        run {
            val keys = ArrayList(views.keys)
            val stride = maxOf(1, keys.size / 6000)
            var idx = 0
            var i = 0
            while (i < keys.size && idx < born.size) {
                born[idx] = mapFrame(bornMap[keys[i]] ?: 0)
                idx++; i += stride
            }
        }
        val traj2 = FloatArray(3 * tn)
        val fv = IntArray(tn); val fb = FloatArray(tn); val fp = FloatArray(tn)
        for (t in 0 until tn) {
            val src = (t * thumbStride).coerceAtMost(n - 1)
            traj2[3*t] = s.traj.getOrElse(3*src) { 0f }
            traj2[3*t+1] = s.traj.getOrElse(3*src+1) { 0f }
            traj2[3*t+2] = s.traj.getOrElse(3*src+2) { 0f }
            fv[t] = statVoxels.getOrElse(src) { views.size }
            fb[t] = statBlind.getOrElse(src) { s.blindPct }
            fp[t] = statPath.getOrElse(src) { pathMm }
        }
        val les = s.lesions.map { StudyEngine.Lesion(it.x, it.y, it.z, it.sizeMm,
            it.support, it.avgConf, mapFrame(it.frame)) }
        StudyEngine.StudyResult(s.pts, s.colors, s.blind, born, traj2, les,
            tn, views.size, s.blindPct, pathMm, detFrames, totalSecs,
            fv, fb, fp, ArrayList(thumbs), stageSecs.copyOf(), s.viewsArr, s.spanMm,
            s.mesh, floatArrayOf(s.cX, s.cY, s.cZ))
    }
}
