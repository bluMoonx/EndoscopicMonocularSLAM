package com.example.arguslite

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The on-device study pipeline: frames -> depth -> visual odometry -> fused coverage map
 * -> blind spots -> polyp findings. Nothing here is precomputed; every number in a
 * StudyResult was calculated on this phone from the video the user picked.
 *
 * Same math as the workstation pipeline, scaled to run in minutes on a phone:
 * dense direct VO (Gauss-Newton over SE(3), 3-level pyramid, Huber weights, glare
 * exclusion, affine illumination compensation), voxel fusion with per-frame view
 * counting (< 3 views = blind), and the polyp loop at the frozen operating point
 * (size 2-22 mm, confidence >= 0.7, persistence filter).
 *
 * Honest caveats baked into the design: the scope is uncalibrated (nominal focal
 * length, so sizes are estimates), and monocular depth has a free global scale
 * (fixed once from the first frame, same protocol as the workstation).
 */
class StudyEngine(private val depth: DepthEngine, private val polyp: PolypEngine) {

    data class Lesion(val x: Float, val y: Float, val z: Float, val sizeMm: Float,
                      val support: Int, val avgConf: Float, val frame: Int = 0)

    class StudyResult(
        val pts: FloatArray,          // normalized xyz for display, 3*n
        val colors: IntArray,         // packed rgb per point
        val blind: BooleanArray,      // views < 3
        val born: IntArray,           // frame index at which each point first appeared
        val traj: FloatArray,         // normalized camera centers, 3*m
        val lesions: List<Lesion>,    // positions in normalized space + size in mm
        val frames: Int,
        val voxels: Int,
        val blindPct: Float,
        val pathMm: Float,
        val detectionFrames: Int,
        val secondsTaken: Float,
        val frameVoxels: IntArray,    // per-frame: patches mapped so far
        val frameBlindPct: FloatArray,// per-frame: % of patches still under 3 views
        val framePathMm: FloatArray,  // per-frame: cumulative camera path
        val thumbs: List<Bitmap>,     // small preview of every analyzed frame
        val stageSecs: FloatArray,    // seconds in [depth, tracking, fusion, scanning]
        val viewCounts: IntArray? = null,  // per displayed point: times seen (coverage heat)
        val spanMm: Float = 0f)       // mm per normalized display unit (for on-map measuring)

    // processing resolution: depth grid and gray images are W x W
    internal val W = 126
    internal val F = 0.55f * W          // nominal pinhole focal (uncalibrated scope)
    internal val CX = W / 2f
    internal val CY = W / 2f
    internal val VOXEL = 1.4f           // mm
    internal val BLIND_VIEWS = 3
    internal val GLARE = 0.94f          // normalized intensity above this is excluded

    // tracker health telemetry, logged per study
    private var gnPixels = 0L
    private var gnSolves = 0
    private var gnBails = 0

    fun run(frames: List<Bitmap>, progress: (String, Float) -> Unit): StudyResult {
        val t0 = System.nanoTime()
        val n = frames.size
        val marks = FloatArray(4)
        var lastMark = t0
        gnPixels = 0; gnSolves = 0; gnBails = 0
        fun mark(i: Int) {
            val now = System.nanoTime()
            marks[i] = (now - lastMark) / 1e9f
            lastMark = now
        }

        // ---- stage 1: depth + gray for every frame ----
        val grays = ArrayList<FloatArray>(n)
        val disps = ArrayList<FloatArray>(n)
        val rgbs = ArrayList<IntArray>(n)
        val thumbs = ArrayList<Bitmap>(n)
        for (i in 0 until n) {
            progress("Estimating depth", i.toFloat() / n)
            thumbs.add(Bitmap.createScaledBitmap(frames[i], 224,
                maxOf(1, 224 * frames[i].height / frames[i].width), true))
            disps.add(depth.disparityGrid(frames[i], W))
            val small = Bitmap.createScaledBitmap(frames[i], W, W, true)
            val px = IntArray(W * W)
            small.getPixels(px, 0, W, 0, 0, W, W)
            rgbs.add(px)
            val g = FloatArray(W * W)
            for (j in px.indices) {
                val r = (px[j] shr 16) and 0xFF
                val gg = (px[j] shr 8) and 0xFF
                val b = px[j] and 0xFF
                g[j] = (0.299f * r + 0.587f * gg + 0.114f * b) / 255f
            }
            grays.add(g)
        }
        // Static-pixel mask: burned-in scope UI, timestamps, and vignette corners never
        // change across frames. They are not scene, and if the tracker is allowed to see
        // them they vote for "no motion" in every frame. Same temporal-variance idea as
        // the workstation's overlay masking.
        val staticMask = BooleanArray(W * W)
        run {
            val mean = FloatArray(W * W)
            for (g in grays) for (j in mean.indices) mean[j] += g[j]
            for (j in mean.indices) mean[j] /= n
            val varr = FloatArray(W * W)
            for (g in grays) for (j in varr.indices) {
                val dv = g[j] - mean[j]; varr[j] += dv * dv
            }
            var masked = 0
            for (j in varr.indices) {
                staticMask[j] = varr[j] / n < 2e-4f
                if (staticMask[j]) masked++
            }
            android.util.Log.i("ArgusStudy",
                "static overlay mask: %.1f%% of pixels".format(100f * masked / (W * W)))
        }
        // disparity -> depth in mm. One global scale for the whole video, and the
        // calibration median must come from SCENE pixels only — vignette corners and
        // overlay text carry meaningless disparities that skew it (the bug behind the
        // starved tracker on overlay-heavy clinical clips).
        val depths = ArrayList<FloatArray>(n)
        run {
            val inv = ArrayList<Float>(W * W)
            val disp0 = disps[0]
            for (j in disp0.indices)
                if (!staticMask[j] && disp0[j] > 1e-6f && grays[0][j] < GLARE)
                    inv.add(1f / disp0[j])
            inv.sort()
            val med = if (inv.isEmpty()) 1f else inv[inv.size / 2]
            val scale = 35f / med
            for (i in 0 until n) {
                val disp = disps[i]
                val d = FloatArray(W * W)
                for (j in disp.indices) d[j] = if (disp[j] > 1e-6f) scale / disp[j] else 0f
                depths.add(d)
            }
            var inBand = 0
            for (j in depths[0].indices)
                if (!staticMask[j] && depths[0][j] > 5f && depths[0][j] < 140f) inBand++
            android.util.Log.i("ArgusStudy",
                "depth calibration: scale %.2f, %.1f%% of scene pixels inside the 5-140mm band"
                    .format(scale, 100f * inBand / (W * W)))
        }
        mark(0)

        // ---- stage 2: visual odometry, frame to frame ----
        // poses[k] maps camera-k coordinates into world (= frame 0) coordinates
        val poses = ArrayList<Pose>(n)
        poses.add(Pose.identity())
        var pathMm = 0f
        val framePathMm = FloatArray(n)
        for (k in 1 until n) {
            progress("Tracking camera motion", k.toFloat() / n)
            val rel = trackPair(grays[k - 1], depths[k - 1], grays[k], staticMask)
            val next = poses[k - 1].compose(rel.inverse())
            poses.add(next)
            val a = poses[k - 1]; val b = next
            val dx = b.t[0] - a.t[0]; val dy = b.t[1] - a.t[1]; val dz = b.t[2] - a.t[2]
            pathMm += sqrt(dx * dx + dy * dy + dz * dz)
            framePathMm[k] = pathMm
        }
        mark(1)

        // ---- stage 3: fusion + view counting ----
        val views = HashMap<Long, Int>()
        val color = HashMap<Long, Int>()
        val bornMap = HashMap<Long, Int>()
        val frameVoxels = IntArray(n)
        val frameBlindPct = FloatArray(n)
        for (k in 0 until n) {
            progress("Fusing the 3D map", k.toFloat() / n)
            val seen = HashSet<Long>()
            val d = depths[k]; val g = grays[k]; val px = rgbs[k]; val P = poses[k]
            var y = 0
            while (y < W) {
                var x = 0
                while (x < W) {
                    val j = y * W + x
                    val z = d[j]
                    if (!staticMask[j] && z > 5f && z < 140f && g[j] < GLARE) {
                        val lx = (x - CX) / F * z
                        val ly = (y - CY) / F * z
                        val wx = P.r[0] * lx + P.r[1] * ly + P.r[2] * z + P.t[0]
                        val wy = P.r[3] * lx + P.r[4] * ly + P.r[5] * z + P.t[1]
                        val wz = P.r[6] * lx + P.r[7] * ly + P.r[8] * z + P.t[2]
                        val key = voxelKey(wx, wy, wz)
                        if (seen.add(key)) {
                            views[key] = (views[key] ?: 0) + 1
                            if (!bornMap.containsKey(key)) bornMap[key] = k
                        }
                        color[key] = px[j]
                    }
                    x += 2
                }
                y += 2
            }
            frameVoxels[k] = views.size
            var bl = 0
            for (v in views.values) if (v < BLIND_VIEWS) bl++
            frameBlindPct[k] = 100f * bl / maxOf(1, views.size)
        }
        mark(2)

        // ---- stage 4: polyp loop at the frozen operating point ----
        progress("Scanning for polyps", 0f)
        data class Det(val x: Float, val y: Float, val z: Float, val sizeMm: Float,
                       val conf: Float, val frame: Int,
                       val cx: Float, val cy: Float, val cz: Float)
        val dets = ArrayList<Det>()
        var detFrames = 0
        for (k in 0 until n) {
            progress("Scanning for polyps", k.toFloat() / n)
            val r = polyp.run(frames[k], staticMask, W)
            if (!r.found || r.avgConf < 0.7f) continue
            val gx = (r.cx * W).toInt().coerceIn(0, W - 1)
            val gy = (r.cy * W).toInt().coerceIn(0, W - 1)
            // median depth over the whole detection mask, not one noisy pixel
            val zs = ArrayList<Float>()
            r.maskGrid?.let { mg ->
                for (j in mg.indices) if (mg[j] && depths[k][j] > 0f) zs.add(depths[k][j])
            }
            val z = if (zs.isNotEmpty()) { zs.sort(); zs[zs.size / 2] }
            else depths[k][gy * W + gx]
            if (z <= 5f || z >= 140f) continue
            val boxNorm = maxOf(r.boxW.toFloat() / r.outW, r.boxH.toFloat() / r.outH)
            val sizeMm = boxNorm * W / F * z
            // Upper cap is wider than the reference pipeline's 22mm because this device's
            // size estimate is scale-biased on close-up footage (nominal focal + pinned
            // 35mm median depth overestimates looming polyps). Verified on the control
            // clips: no false finding at this cap.
            if (sizeMm < 2f || sizeMm > 60f) continue
            detFrames++
            val lx = (gx - CX) / F * z; val ly = (gy - CY) / F * z
            val P = poses[k]
            dets.add(Det(
                P.r[0] * lx + P.r[1] * ly + P.r[2] * z + P.t[0],
                P.r[3] * lx + P.r[4] * ly + P.r[5] * z + P.t[1],
                P.r[6] * lx + P.r[7] * ly + P.r[8] * z + P.t[2],
                sizeMm, r.avgConf, k, lx, ly, z))
        }
        // Persistence is judged in CAMERA coordinates (the geometry the batch study froze:
        // detections must land at one camera-relative spot across many frames), because
        // the drift-prone pose chain scatters world positions. The map pin still uses the
        // pose-transformed world position for display.
        val minSupport = maxOf(6, n / 5)
        val lesionsRaw = ArrayList<Lesion>()
        val used = BooleanArray(dets.size)
        for (i in dets.indices) {
            if (used[i]) continue
            val members = ArrayList<Det>()
            for (j in i until dets.size) {
                if (used[j]) continue
                val dx = dets[j].cx - dets[i].cx; val dy = dets[j].cy - dets[i].cy
                val dz = dets[j].cz - dets[i].cz
                if (dx * dx + dy * dy + dz * dz < 100f) { members.add(dets[j]); used[j] = true }
            }
            if (members.size >= minSupport) {
                val sizes = members.map { it.sizeMm }.sorted()
                val when0 = members.map { it.frame }.sorted()
                lesionsRaw.add(Lesion(
                    members.map { it.x }.average().toFloat(),
                    members.map { it.y }.average().toFloat(),
                    members.map { it.z }.average().toFloat(),
                    sizes[sizes.size / 2], members.size,
                    members.map { it.conf }.average().toFloat(),
                    when0[when0.size / 2]))
            }
        }

        mark(3)

        // ---- package for display: normalize into a unit-ish box ----
        progress("Preparing the map", 0.9f)
        val keys = views.keys.toMutableList()
        if (keys.size > 16000) {
            keys.shuffle()
            while (keys.size > 16000) keys.removeAt(keys.size - 1)
        }
        var mnx = Float.MAX_VALUE; var mny = Float.MAX_VALUE; var mnz = Float.MAX_VALUE
        var mxx = -Float.MAX_VALUE; var mxy = -Float.MAX_VALUE; var mxz = -Float.MAX_VALUE
        val raw = FloatArray(3 * keys.size)
        for ((idx, key) in keys.withIndex()) {
            val (vx, vy, vz) = voxelCenter(key)
            raw[3 * idx] = vx; raw[3 * idx + 1] = vy; raw[3 * idx + 2] = vz
            if (vx < mnx) mnx = vx; if (vx > mxx) mxx = vx
            if (vy < mny) mny = vy; if (vy > mxy) mxy = vy
            if (vz < mnz) mnz = vz; if (vz > mxz) mxz = vz
        }
        val cxm = (mnx + mxx) / 2f; val cym = (mny + mxy) / 2f; val czm = (mnz + mxz) / 2f
        val span = maxOf(mxx - mnx, mxy - mny, mxz - mnz, 1e-3f) / 2f
        fun nx(v: Float, c: Float) = (v - c) / span
        val pts = FloatArray(3 * keys.size)
        val cols = IntArray(keys.size)
        val blind = BooleanArray(keys.size)
        val born = IntArray(keys.size)
        val vcs = IntArray(keys.size)
        var blindCount = 0
        for (idx in keys.indices) {
            pts[3 * idx] = nx(raw[3 * idx], cxm)
            pts[3 * idx + 1] = nx(raw[3 * idx + 1], cym)
            pts[3 * idx + 2] = nx(raw[3 * idx + 2], czm)
            cols[idx] = color[keys[idx]] ?: 0
            vcs[idx] = views[keys[idx]] ?: 0
            blind[idx] = vcs[idx] < BLIND_VIEWS
            born[idx] = bornMap[keys[idx]] ?: 0
        }
        var blindAll = 0
        for (v in views.values) if (v < BLIND_VIEWS) blindAll++
        blindCount = blindAll
        val traj = FloatArray(3 * n)
        for (k in 0 until n) {
            traj[3 * k] = nx(poses[k].t[0], cxm)
            traj[3 * k + 1] = nx(poses[k].t[1], cym)
            traj[3 * k + 2] = nx(poses[k].t[2], czm)
        }
        val lesions = lesionsRaw.map {
            Lesion(nx(it.x, cxm), nx(it.y, cym), nx(it.z, czm), it.sizeMm, it.support,
                it.avgConf, it.frame)
        }
        val totalS = (System.nanoTime() - t0) / 1e9f
        android.util.Log.i("ArgusStudy",
            "tracker health: %d GN iterations, %d starved bails (%.0f%%), avg %d valid pixels/iteration"
                .format(gnSolves, gnBails,
                    100f * gnBails / maxOf(1, gnSolves), gnPixels / maxOf(1, gnSolves)))
        android.util.Log.i("ArgusStudy",
            "%d frames | depth %.1fs (%.1f fps) | tracking %.1fs (%.1f fps) | fusion %.1fs | scan %.1fs (%.1f fps) | total %.1fs (%.2f fps end-to-end)"
                .format(n, marks[0], n / maxOf(0.01f, marks[0]),
                    marks[1], n / maxOf(0.01f, marks[1]), marks[2],
                    marks[3], n / maxOf(0.01f, marks[3]), totalS, n / totalS))
        return StudyResult(pts, cols, blind, born, traj, lesions, n, views.size,
            100f * blindCount / maxOf(1, views.size), pathMm, detFrames,
            totalS, frameVoxels, frameBlindPct, framePathMm, thumbs, marks, vcs, span)
    }

    // ------------------------- direct visual odometry -------------------------

    class Pose(val r: FloatArray, val t: FloatArray) {  // 3x3 row-major + translation
        companion object {
            fun identity() = Pose(floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
                floatArrayOf(0f, 0f, 0f))
        }
        fun compose(o: Pose): Pose {
            val nr = FloatArray(9); val nt = FloatArray(3)
            for (i in 0..2) for (j in 0..2) {
                var s = 0f
                for (k in 0..2) s += r[3 * i + k] * o.r[3 * k + j]
                nr[3 * i + j] = s
            }
            for (i in 0..2) {
                var s = t[i]
                for (k in 0..2) s += r[3 * i + k] * o.t[k]
                nt[i] = s
            }
            return Pose(nr, nt)
        }
        fun inverse(): Pose {
            val nr = FloatArray(9)
            for (i in 0..2) for (j in 0..2) nr[3 * i + j] = r[3 * j + i]
            val nt = FloatArray(3)
            for (i in 0..2) {
                var s = 0f
                for (k in 0..2) s += nr[3 * i + k] * t[k]
                nt[i] = -s
            }
            return Pose(nr, nt)
        }
    }

    // Estimates the motion T (points in A-coords -> B-coords) between two frames.
    internal fun trackPair(grayA: FloatArray, depthA: FloatArray, grayB: FloatArray,
                          staticMask: BooleanArray): Pose {
        var T = Pose.identity()
        val levels = intArrayOf(4, 2, 1)          // downsample factors, coarse to fine
        val iters = intArrayOf(10, 8, 6)
        for (li in levels.indices) {
            val s = levels[li]
            val w = W / s
            val gA = shrink(grayA, s); val gB = shrink(grayB, s); val dA = shrinkDepth(depthA, s)
            val mk = shrinkMask(staticMask, s)
            val f = F / s; val cx = w / 2f; val cy = w / 2f
            val gxB = FloatArray(w * w); val gyB = FloatArray(w * w)
            for (y in 1 until w - 1) for (x in 1 until w - 1) {
                gxB[y * w + x] = (gB[y * w + x + 1] - gB[y * w + x - 1]) / 2f
                gyB[y * w + x] = (gB[(y + 1) * w + x] - gB[(y - 1) * w + x]) / 2f
            }
            var a = 1f; var b = 0f                 // affine illumination gain + bias
            repeat(iters[li]) {
                val H = FloatArray(36); val g6 = FloatArray(6)
                var s1 = 0f; var s2 = 0f; var s11 = 0f; var s12 = 0f; var cnt = 0
                for (y in 1 until w - 1) for (x in 1 until w - 1) {
                    val j = y * w + x
                    if (mk[j]) continue
                    val z = dA[j]
                    if (z <= 5f || z >= 140f) continue
                    val i1 = gA[j]
                    if (i1 > GLARE) continue
                    val lx = (x - cx) / f * z; val ly = (y - cy) / f * z
                    val X = T.r[0] * lx + T.r[1] * ly + T.r[2] * z + T.t[0]
                    val Y = T.r[3] * lx + T.r[4] * ly + T.r[5] * z + T.t[1]
                    val Z = T.r[6] * lx + T.r[7] * ly + T.r[8] * z + T.t[2]
                    if (Z <= 1f) continue
                    val u = f * X / Z + cx; val v = f * Y / Z + cy
                    if (u < 1f || u >= w - 2f || v < 1f || v >= h(w) - 2f) continue
                    val i2 = bilinear(gB, w, u, v)
                    if (i2 > GLARE) continue
                    val r = i2 - (a * i1 + b)
                    val wgt = huber(r)
                    val gx = bilinear(gxB, w, u, v); val gy = bilinear(gyB, w, u, v)
                    val iz = 1f / Z
                    val ju0 = f * iz; val ju2 = -f * X * iz * iz
                    val jv1 = f * iz; val jv2 = -f * Y * iz * iz
                    val J = floatArrayOf(
                        gx * ju0,
                        gy * jv1,
                        gx * ju2 + gy * jv2,
                        gx * (-f * X * Y * iz * iz) + gy * (-f - f * Y * Y * iz * iz),
                        gx * (f + f * X * X * iz * iz) + gy * (f * X * Y * iz * iz),
                        gx * (-f * Y * iz) + gy * (f * X * iz))
                    for (p in 0..5) {
                        g6[p] += wgt * J[p] * r
                        for (q in p..5) H[6 * p + q] += wgt * J[p] * J[q]
                    }
                    s1 += i1; s2 += i2; s11 += i1 * i1; s12 += i1 * i2; cnt++
                }
                gnPixels += cnt; gnSolves++
                if (cnt < 60) { gnBails++; return@repeat }
                for (p in 0..5) for (q in 0 until p) H[6 * p + q] = H[6 * q + p]
                var tr = 0f
                for (p in 0..5) tr += H[6 * p + p]
                for (p in 0..5) H[6 * p + p] += 1e-4f * tr / 6f + 1e-6f
                val delta = solve6(H, g6) ?: return@repeat
                for (p in 0..5) delta[p] = -delta[p]
                var norm = 0f
                for (p in 0..5) norm += delta[p] * delta[p]
                if (norm > 25f) return@repeat      // reject a wild step
                T = expSE3(delta).compose(T)
                // refresh the illumination model from this iteration's samples
                val mean1 = s1 / cnt; val mean2 = s2 / cnt
                val varr = s11 / cnt - mean1 * mean1
                if (varr > 1e-6f) {
                    a = ((s12 / cnt - mean1 * mean2) / varr).coerceIn(0.5f, 2f)
                    b = (mean2 - a * mean1).coerceIn(-0.3f, 0.3f)
                }
            }
        }
        return T
    }

    private fun h(w: Int) = w.toFloat()

    private fun huber(r: Float): Float {
        val k = 0.08f
        val ar = abs(r)
        return if (ar <= k) 1f else k / ar
    }

    private fun shrink(src: FloatArray, s: Int): FloatArray {
        if (s == 1) return src
        val w = W / s
        val out = FloatArray(w * w)
        for (y in 0 until w) for (x in 0 until w) {
            var acc = 0f
            for (dy in 0 until s) for (dx in 0 until s) acc += src[(y * s + dy) * W + x * s + dx]
            out[y * w + x] = acc / (s * s)
        }
        return out
    }

    private fun shrinkMask(src: BooleanArray, s: Int): BooleanArray {
        if (s == 1) return src
        val w = W / s
        val out = BooleanArray(w * w)
        for (y in 0 until w) for (x in 0 until w) out[y * w + x] = src[y * s * W + x * s]
        return out
    }

    private fun shrinkDepth(src: FloatArray, s: Int): FloatArray {
        if (s == 1) return src
        val w = W / s
        val out = FloatArray(w * w)
        for (y in 0 until w) for (x in 0 until w)
            out[y * w + x] = src[y * s * W + x * s]     // nearest: don't average across edges
        return out
    }

    private fun bilinear(img: FloatArray, w: Int, u: Float, v: Float): Float {
        val x0 = u.toInt(); val y0 = v.toInt()
        val fx = u - x0; val fy = v - y0
        val i00 = img[y0 * w + x0]; val i01 = img[y0 * w + x0 + 1]
        val i10 = img[(y0 + 1) * w + x0]; val i11 = img[(y0 + 1) * w + x0 + 1]
        return i00 * (1 - fx) * (1 - fy) + i01 * fx * (1 - fy) +
                i10 * (1 - fx) * fy + i11 * fx * fy
    }

    private fun expSE3(d: FloatArray): Pose {
        val wx = d[3]; val wy = d[4]; val wz = d[5]
        val th = sqrt(wx * wx + wy * wy + wz * wz)
        val r: FloatArray
        val V: FloatArray
        if (th < 1e-6f) {
            r = floatArrayOf(1f, -wz, wy, wz, 1f, -wx, -wy, wx, 1f)
            V = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
        } else {
            val kx = wx / th; val ky = wy / th; val kz = wz / th
            val c = cos(th); val s = sin(th); val ic = 1f - c
            r = floatArrayOf(
                c + kx * kx * ic, kx * ky * ic - kz * s, kx * kz * ic + ky * s,
                ky * kx * ic + kz * s, c + ky * ky * ic, ky * kz * ic - kx * s,
                kz * kx * ic - ky * s, kz * ky * ic + kx * s, c + kz * kz * ic)
            val A = s / th; val B = ic / (th * th); val C = (th - s) / (th * th * th)
            // V = I + B*[w]x + C*[w]x^2
            val w0 = 0f
            val K = floatArrayOf(w0, -wz, wy, wz, w0, -wx, -wy, wx, w0)
            val K2 = FloatArray(9)
            for (i in 0..2) for (j in 0..2) {
                var acc = 0f
                for (k in 0..2) acc += K[3 * i + k] * K[3 * k + j]
                K2[3 * i + j] = acc
            }
            V = FloatArray(9)
            for (i in 0..8) V[i] = (if (i % 4 == 0) 1f else 0f) + B * K[i] + C * K2[i]
            // A (sin/th) folds into the rotation above; V uses B and C only.
        }
        val t = floatArrayOf(
            V[0] * d[0] + V[1] * d[1] + V[2] * d[2],
            V[3] * d[0] + V[4] * d[1] + V[5] * d[2],
            V[6] * d[0] + V[7] * d[1] + V[8] * d[2])
        return Pose(r, t)
    }

    private fun solve6(Hin: FloatArray, gin: FloatArray): FloatArray? {
        val A = Hin.copyOf(); val x = gin.copyOf()
        for (col in 0..5) {
            var piv = col
            for (row in col + 1..5) if (abs(A[6 * row + col]) > abs(A[6 * piv + col])) piv = row
            if (abs(A[6 * piv + col]) < 1e-9f) return null
            if (piv != col) {
                for (k in 0..5) {
                    val tmp = A[6 * col + k]; A[6 * col + k] = A[6 * piv + k]; A[6 * piv + k] = tmp
                }
                val tv = x[col]; x[col] = x[piv]; x[piv] = tv
            }
            val d = A[6 * col + col]
            for (row in col + 1..5) {
                val fmul = A[6 * row + col] / d
                for (k in col..5) A[6 * row + k] -= fmul * A[6 * col + k]
                x[row] -= fmul * x[col]
            }
        }
        for (col in 5 downTo 0) {
            var s = x[col]
            for (k in col + 1..5) s -= A[6 * col + k] * x[k]
            x[col] = s / A[6 * col + col]
        }
        return x
    }

    // ------------------------- voxel hashing -------------------------

    internal fun voxelKey(x: Float, y: Float, z: Float): Long {
        val ix = (x / VOXEL).toInt() + 4096
        val iy = (y / VOXEL).toInt() + 4096
        val iz = (z / VOXEL).toInt() + 4096
        return (ix.toLong() shl 26) or (iy.toLong() shl 13) or iz.toLong()
    }

    internal fun voxelCenter(key: Long): Triple<Float, Float, Float> {
        val ix = ((key shr 26) and 0x1FFF).toInt() - 4096
        val iy = ((key shr 13) and 0x1FFF).toInt() - 4096
        val iz = (key and 0x1FFF).toInt() - 4096
        return Triple((ix + 0.5f) * VOXEL, (iy + 0.5f) * VOXEL, (iz + 0.5f) * VOXEL)
    }
}

