package com.example.arguslite

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** Direct visual odometry, extracted from StudyEngine so the exact shipped math is
 *  callable from a plain JVM unit test (no Android Context needed).
 *
 *  Two residual-weighting modes, per the pre-registered port trial
 *  (results/huber_port_gate.md):
 *   - FIXED_K: the weighting the app has always shipped (Huber with a fixed
 *     0.08 intensity threshold). Bit-identical to the pre-extraction code.
 *   - MEDIAN_SCALED: the workstation's adopted pipeline-v2 kernel
 *     (s = 1.4826*median|r| + 1e-6; a = |r|/(1.345 s); w = min(1, 1/a)),
 *     computed in two passes per Gauss-Newton iteration.
 *  The shipped default stays FIXED_K unless the trial's gates P, A and S all pass. */
internal object Vo {
    const val W = 126
    val F = 0.55f * W                   // nominal pinhole focal (uncalibrated scope)
    const val GLARE = 0.94f             // normalized intensity above this is excluded

    enum class Mode { FIXED_K, MEDIAN_SCALED }
    @Volatile var mode = Mode.FIXED_K   // process-wide default; flipped only by adoption

    var gnPixels = 0L
    var gnSolves = 0
    var gnBails = 0
    fun resetStats() { gnPixels = 0; gnSolves = 0; gnBails = 0 }

    // Estimates the motion T (points in A-coords -> B-coords) between two frames.
    fun trackPair(grayA: FloatArray, depthA: FloatArray, grayB: FloatArray,
                  staticMask: BooleanArray, m: Mode = mode): StudyEngine.Pose {
        var T = StudyEngine.Pose.identity()
        val levels = intArrayOf(4, 2, 1)          // downsample factors, coarse to fine
        val iters = intArrayOf(10, 8, 6)
        val rbuf = if (m == Mode.MEDIAN_SCALED) FloatArray(W * W) else FloatArray(0)
        val jbuf = if (m == Mode.MEDIAN_SCALED) FloatArray(W * W * 6) else FloatArray(0)
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
                    if (m == Mode.FIXED_K) {
                        val wgt = huber(r)
                        for (p in 0..5) {
                            g6[p] += wgt * J[p] * r
                            for (q in p..5) H[6 * p + q] += wgt * J[p] * J[q]
                        }
                    } else {
                        rbuf[cnt] = r
                        for (p in 0..5) jbuf[cnt * 6 + p] = J[p]
                    }
                    s1 += i1; s2 += i2; s11 += i1 * i1; s12 += i1 * i2; cnt++
                }
                gnPixels += cnt; gnSolves++
                if (cnt < 60) { gnBails++; return@repeat }
                if (m == Mode.MEDIAN_SCALED) {
                    // pass 2: weight by the median-scaled Huber kernel, then accumulate
                    val wts = medianScaledWeights(rbuf, cnt)
                    for (i in 0 until cnt) {
                        val wgt = wts[i]; val r = rbuf[i]
                        for (p in 0..5) {
                            val jp = jbuf[i * 6 + p]
                            g6[p] += wgt * jp * r
                            for (q in p..5) H[6 * p + q] += wgt * jp * jbuf[i * 6 + q]
                        }
                    }
                }
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

    /** Pipeline-v2 weights for the first n residuals in r: s = 1.4826*median|r| + 1e-6,
     *  a = |r| / (1.345 s), w = min(1, 1/a). Matches direct_vo.py:93-96 including the
     *  even-count median (average of the two middle values). */
    fun medianScaledWeights(r: FloatArray, n: Int): FloatArray {
        val absr = FloatArray(n)
        for (i in 0 until n) absr[i] = abs(r[i])
        val sorted = absr.copyOf()
        java.util.Arrays.sort(sorted)
        val med = if (n % 2 == 1) sorted[n / 2]
                  else 0.5f * (sorted[n / 2 - 1] + sorted[n / 2])
        val s = 1.4826f * med + 1e-6f
        val out = FloatArray(n)
        for (i in 0 until n) {
            val a = absr[i] / (1.345f * s)
            out[i] = if (a <= 1f) 1f else 1f / a
        }
        return out
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

    private fun expSE3(d: FloatArray): StudyEngine.Pose {
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
        return StudyEngine.Pose(r, t)
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
}
