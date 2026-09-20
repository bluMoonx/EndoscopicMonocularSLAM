package com.example.arguslite

import android.graphics.Paint
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.nativeCanvas
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The surface layer: turns the fused voxel volume into an actual lit 3D surface,
 * the way desktop reconstruction viewers do, instead of a cloud of dots.
 *
 * - During fusion the engines accumulate a truncated signed distance per voxel
 *   (positive in open space toward the camera, negative just behind the wall).
 * - buildSurface() extracts the zero-crossing as a triangle mesh using surface
 *   nets: one vertex per sign-change cell, quads across sign-change edges.
 * - DrawScope.drawMeshLit() renders the sorted, head-lit triangles in a single
 *   native drawVertices call.
 */

const val TSDF_TRUNC_VOXELS = 2.5f

fun tsdfPack(sum: Float, weight: Float): Long =
    (java.lang.Float.floatToRawIntBits(sum).toLong() shl 32) or
            (java.lang.Float.floatToRawIntBits(weight).toLong() and 0xFFFFFFFFL)

fun tsdfSum(p: Long): Float = java.lang.Float.intBitsToFloat((p ushr 32).toInt())
fun tsdfWeight(p: Long): Float = java.lang.Float.intBitsToFloat((p and 0xFFFFFFFFL).toInt())

class MeshData(val verts: FloatArray,   // world mm, 3 per vertex
               val norms: FloatArray,   // 3 per vertex, unit-ish
               val baseCol: IntArray,   // packed rgb tissue color per vertex
               val views: IntArray,     // coverage count per vertex
               val tris: IntArray)      // 3 vertex indices per triangle

/** Load a workstation-exported mesh (assets/mesh_*.bin):
 *  int32 nv, int32 nt, f32 verts[3nv] mm, f32 norms[3nv], u8 rgb[3nv],
 *  u8 views[nv], int32 tris[3nt]. */
fun loadMeshBin(bytes: ByteArray): MeshData {
    val bb = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    val nv = bb.int; val nt = bb.int
    val verts = FloatArray(3 * nv) { bb.float }
    val norms = FloatArray(3 * nv) { bb.float }
    val cols = IntArray(nv)
    for (i in 0 until nv) {
        val r = bb.get().toInt() and 0xFF
        val g = bb.get().toInt() and 0xFF
        val b = bb.get().toInt() and 0xFF
        cols[i] = (r shl 16) or (g shl 8) or b
    }
    val views = IntArray(nv) { bb.get().toInt() and 0xFF }
    val tris = IntArray(3 * nt) { bb.int }
    return MeshData(verts, norms, cols, views, tris)
}

fun meshCenterSpan(m: MeshData): FloatArray {
    var mn = floatArrayOf(Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE)
    var mx = floatArrayOf(-Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE)
    var i = 0
    while (i < m.verts.size) {
        for (a in 0..2) {
            if (m.verts[i + a] < mn[a]) mn[a] = m.verts[i + a]
            if (m.verts[i + a] > mx[a]) mx[a] = m.verts[i + a]
        }
        i += 3
    }
    val span = maxOf(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-3f) / 2f
    return floatArrayOf((mn[0] + mx[0]) / 2f, (mn[1] + mx[1]) / 2f,
        (mn[2] + mx[2]) / 2f, span)
}

private val DX = 1L shl 26
private val DY = 1L shl 13
private val DZ = 1L

fun buildSurface(tsdf: HashMap<Long, Long>,
                 color: HashMap<Long, Int>,
                 viewCount: HashMap<Long, Int>,
                 center: (Long) -> Triple<Float, Float, Float>): MeshData {
    fun sdfAt(k: Long): Float? {
        val p = tsdf[k] ?: return null
        val w = tsdfWeight(p)
        return if (w > 0f) tsdfSum(p) / w else null
    }
    val verts = ArrayList<Float>()
    val norms = ArrayList<Float>()
    val cols = ArrayList<Int>()
    val vviews = ArrayList<Int>()
    val tris = ArrayList<Int>()
    val cellVert = HashMap<Long, Int>()

    fun vertexFor(cell: Long): Int {
        cellVert[cell]?.let { return it }
        // corner samples of the cell; a missing corner disqualifies the cell
        val ck = longArrayOf(cell, cell + DX, cell + DY, cell + DX + DY,
            cell + DZ, cell + DX + DZ, cell + DY + DZ, cell + DX + DY + DZ)
        val s = FloatArray(8)
        for (i in 0..7) s[i] = sdfAt(ck[i]) ?: return -1
        // cell edges as corner-index pairs
        val e = intArrayOf(0,1, 2,3, 4,5, 6,7, 0,2, 1,3, 4,6, 5,7, 0,4, 1,5, 2,6, 3,7)
        var px = 0f; var py = 0f; var pz = 0f; var cnt = 0
        for (i in 0 until 12) {
            val a = e[2 * i]; val b = e[2 * i + 1]
            if ((s[a] < 0f) != (s[b] < 0f)) {
                val t = s[a] / (s[a] - s[b])
                val (ax, ay, az) = center(ck[a])
                val (bx, by, bz) = center(ck[b])
                px += ax + t * (bx - ax); py += ay + t * (by - ay); pz += az + t * (bz - az)
                cnt++
            }
        }
        if (cnt == 0) return -1
        val idx = verts.size / 3
        verts.add(px / cnt); verts.add(py / cnt); verts.add(pz / cnt)
        norms.add(0f); norms.add(0f); norms.add(0f)
        // tissue color / coverage from whatever corners carry data
        var r = 0; var g = 0; var b2 = 0; var nc = 0; var vv = 0; var nv = 0
        for (k in ck) {
            color[k]?.let { r += (it shr 16) and 0xFF; g += (it shr 8) and 0xFF
                b2 += it and 0xFF; nc++ }
            viewCount[k]?.let { vv += it; nv++ }
        }
        cols.add(if (nc > 0) ((r / nc) shl 16) or ((g / nc) shl 8) or (b2 / nc)
        else 0xB07060)
        vviews.add(if (nv > 0) vv / nv else 1)
        cellVert[cell] = idx
        return idx
    }

    for ((s, packed) in tsdf) {
        val w0 = tsdfWeight(packed)
        if (w0 <= 0f) continue
        val s0 = tsdfSum(packed) / w0
        for (axis in 0..2) {
            val d = when (axis) { 0 -> DX; 1 -> DY; else -> DZ }
            val s1 = sdfAt(s + d) ?: continue
            if ((s0 < 0f) == (s1 < 0f)) continue
            val (u, v) = when (axis) { 0 -> DY to DZ; 1 -> DX to DZ; else -> DX to DY }
            // ring of the four cells sharing this edge
            val c0 = vertexFor(s - u - v); val c1 = vertexFor(s - v)
            val c2 = vertexFor(s);         val c3 = vertexFor(s - u)
            if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) continue
            if (s0 < 0f) { tris.add(c0); tris.add(c1); tris.add(c2)
                tris.add(c0); tris.add(c2); tris.add(c3) }
            else { tris.add(c0); tris.add(c2); tris.add(c1)
                tris.add(c0); tris.add(c3); tris.add(c2) }
        }
    }
    // accumulate face normals into vertices
    val nv = FloatArray(norms.size)
    var i = 0
    while (i < tris.size) {
        val a = tris[i] * 3; val b = tris[i + 1] * 3; val c = tris[i + 2] * 3
        val ux = verts[b] - verts[a]; val uy = verts[b+1] - verts[a+1]; val uz = verts[b+2] - verts[a+2]
        val vx = verts[c] - verts[a]; val vy = verts[c+1] - verts[a+1]; val vz = verts[c+2] - verts[a+2]
        val nx = uy * vz - uz * vy; val ny = uz * vx - ux * vz; val nz = ux * vy - uy * vx
        for (o in intArrayOf(a, b, c)) { nv[o] += nx; nv[o+1] += ny; nv[o+2] += nz }
        i += 3
    }
    i = 0
    while (i < nv.size) {
        val len = sqrt(nv[i]*nv[i] + nv[i+1]*nv[i+1] + nv[i+2]*nv[i+2])
        if (len > 1e-6f) { nv[i] /= len; nv[i+1] /= len; nv[i+2] /= len }
        i += 3
    }
    return MeshData(verts.toFloatArray(), nv, cols.toIntArray(), vviews.toIntArray(),
        tris.toIntArray())
}

/** Render the mesh head-lit: far triangles first, one native drawVertices call.
 *  mode: 0 tissue colors, 1 coverage heat, 2 blind highlight. */
fun DrawScope.drawMeshLit(mesh: MeshData, cX: Float, cY: Float, cZ: Float, span: Float,
                          yaw: Float, pitch: Float, zoom: Float, mode: Int,
                          blindBelow: Int) {
    val n = mesh.verts.size / 3
    if (n == 0 || mesh.tris.isEmpty()) return
    val cyw = cos(yaw); val syw = sin(yaw)
    val cp = cos(pitch); val sp = sin(pitch)
    val s = minOf(size.width, size.height) * 0.42f
    val px = FloatArray(n); val py = FloatArray(n); val depth = FloatArray(n)
    val lit = FloatArray(n)
    for (i in 0 until n) {
        val x = (mesh.verts[3*i] - cX) / span
        val y = (mesh.verts[3*i+1] - cY) / span
        val z = (mesh.verts[3*i+2] - cZ) / span
        val x1 = x * cyw + z * syw; val z1 = -x * syw + z * cyw
        val y1 = y * cp - z1 * sp; val z2 = y * sp + z1 * cp
        val persp = zoom / (1.6f - z2 * 0.5f)
        px[i] = size.width / 2f + x1 * s * persp
        py[i] = size.height / 2f + y1 * s * persp
        depth[i] = z2
        // headlight: normal's view-space z component
        val nx = mesh.norms[3*i]; val ny = mesh.norms[3*i+1]; val nz = mesh.norms[3*i+2]
        val nz1 = -nx * syw + nz * cyw
        val nz2 = ny * sp + nz1 * cp
        lit[i] = 0.30f + 0.70f * kotlin.math.abs(nz2)
    }
    val t = mesh.tris.size / 3
    val order = (0 until t).sortedBy {
        depth[mesh.tris[3*it]] + depth[mesh.tris[3*it+1]] + depth[mesh.tris[3*it+2]]
    }
    val pos = FloatArray(6 * t)
    val col = IntArray(3 * t)
    var o = 0
    for (ti in order) {
        for (k in 0..2) {
            val vi = mesh.tris[3 * ti + k]
            pos[2 * o] = px[vi]; pos[2 * o + 1] = py[vi]
            val li = lit[vi]
            col[o] = when (mode) {
                1 -> {
                    val tt = minOf(mesh.views[vi], 8) / 8f
                    val c = coverageColor(tt, 1f)
                    (0xFF shl 24) or ((c.red * 255 * li).toInt() shl 16) or
                            ((c.green * 255 * li).toInt() shl 8) or (c.blue * 255 * li).toInt()
                }
                2 -> if (mesh.views[vi] < blindBelow)
                    (0xFF shl 24) or ((230 * li).toInt() shl 16) or
                            ((58 * li).toInt() shl 8) or (70 * li).toInt()
                else (0xFF shl 24) or ((150 * li).toInt() shl 16) or
                        ((140 * li).toInt() shl 8) or (135 * li).toInt()
                else -> {
                    val b = mesh.baseCol[vi]
                    (0xFF shl 24) or
                            ((((b shr 16) and 0xFF) * li).toInt() shl 16) or
                            ((((b shr 8) and 0xFF) * li).toInt() shl 8) or
                            ((b and 0xFF) * li).toInt()
                }
            }
            o++
        }
    }
    drawContext.canvas.nativeCanvas.drawVertices(
        android.graphics.Canvas.VertexMode.TRIANGLES, pos.size, pos, 0,
        null, 0, col, 0, null, 0, 0, Paint())
}
