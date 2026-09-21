package com.example.arguslite

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

class DepthEngine(context: Context) {
    private val sz = 252
    private val env = OrtEnvironment.getEnvironment()
    private val session = run {
        // prefer the accuracy-verified quantized model when shipped; try hardware
        // acceleration first and fall back to plain CPU without complaint
        val bytes = runCatching { context.assets.open("depth_q.onnx").readBytes() }
            .getOrElse { context.assets.open("depth.onnx").readBytes() }
        val accelerated = runCatching {
            val o = ai.onnxruntime.OrtSession.SessionOptions()
            o.setIntraOpNumThreads(4)
            o.addNnapi()
            env.createSession(bytes, o)
        }.getOrNull()
        if (accelerated != null) {
            android.util.Log.i("ArgusStudy", "depth session: NNAPI accelerated")
            accelerated
        } else {
            android.util.Log.i("ArgusStudy", "depth session: CPU")
            env.createSession(bytes,
                ai.onnxruntime.OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })
        }
    }
    var lastGrid: FloatArray? = null
    val gridN = 42

    fun infer(frame: Bitmap): Array<FloatArray> {
        val s = Bitmap.createScaledBitmap(frame, sz, sz, true)
        val px = IntArray(sz * sz)
        s.getPixels(px, 0, sz, 0, 0, sz, sz)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val buf = FloatBuffer.allocate(3 * sz * sz)
        for (c in 0..2) for (i in px.indices) {
            val v = when (c) {
                0 -> (px[i] shr 16) and 0xFF   // red
                1 -> (px[i] shr 8) and 0xFF    // green
                else -> px[i] and 0xFF         // blue
            }
            buf.put(((v / 255f) - mean[c]) / std[c])
        }
        buf.rewind()

        val input = OnnxTensor.createTensor(env, buf, longArrayOf(1L, 3L, sz.toLong(), sz.toLong()))
        return session.run(mapOf("pixels" to input)).use { out ->
            (out[0].value as Array<Array<FloatArray>>)[0]
        }
    }

    // Raw disparity pooled to n x n, for the on-device study pipeline.
    fun disparityGrid(frame: Bitmap, n: Int): FloatArray {
        val disp = infer(frame)
        val g = FloatArray(n * n)
        val k = sz / n
        for (gy in 0 until n) for (gx in 0 until n) {
            var s2 = 0f
            for (dy in 0 until k) for (dx in 0 until k) s2 += disp[gy * k + dy][gx * k + dx]
            g[gy * n + gx] = s2 / (k * k)
        }
        return g
    }

    fun run(frame: Bitmap): Bitmap {
        val disp = infer(frame)

        var lo = Float.MAX_VALUE; var hi = -Float.MAX_VALUE
        for (row in disp) for (v in row) { if (v < lo) lo = v; if (v > hi) hi = v }
        val range = (hi - lo).coerceAtLeast(1e-6f)
        val g = FloatArray(gridN * gridN)
        for (gy in 0 until gridN) for (gx in 0 until gridN)
            g[gy * gridN + gx] = (disp[gy * sz / gridN][gx * sz / gridN] - lo) / range
        lastGrid = g

        val heat = IntArray(sz * sz)
        for (y in 0 until sz) for (x in 0 until sz) {
            val t = (disp[y][x] - lo) / range        // 0 = far, 1 = near
            heat[y * sz + x] = colorOf(t)
        }
        return Bitmap.createBitmap(heat, sz, sz, Bitmap.Config.ARGB_8888)
    }

    private fun colorOf(t: Float): Int {
        // dark violet (far) -> red -> yellow (near), like the workstation's heat map
        val r = (255 * minOf(1f, t * 2f)).toInt()
        val g = (255 * maxOf(0f, t * 2f - 1f)).toInt()
        val b = (90 * maxOf(0f, 1f - t * 2.5f)).toInt()
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
}
