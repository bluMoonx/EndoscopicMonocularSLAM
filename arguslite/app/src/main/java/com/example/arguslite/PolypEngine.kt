package com.example.arguslite

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.nio.FloatBuffer

class PolypEngine(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(
        context.assets.open("polyp.onnx").readBytes(),
        OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) })

    data class Result(val overlay: Bitmap, val found: Boolean, val areaPct: Float,
                      val avgConf: Float, val peakConf: Float, val boxW: Int, val boxH: Int,
                      val cx: Float = 0.5f, val cy: Float = 0.5f, val outW: Int = 1,
                      val outH: Int = 1, val maskGrid: BooleanArray? = null)

    // exclude126: static-overlay mask on a 126x126 grid — burned-in scope UI is not tissue
    // and must not count as a detection. gridN > 0 additionally returns the segmentation
    // mask resampled to gridN x gridN, for depth lookups over the whole mask.
    fun run(src: Bitmap, exclude126: BooleanArray? = null, gridN: Int = 0): Result {
        val inW = 384; val inH = 288
        val s = Bitmap.createScaledBitmap(src, inW, inH, true)
        val px = IntArray(inW * inH)
        s.getPixels(px, 0, inW, 0, 0, inW, inH)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val buf = FloatBuffer.allocate(3 * inW * inH)
        for (c in 0..2) for (i in px.indices) {
            val v = when (c) {
                0 -> (px[i] shr 16) and 0xFF
                1 -> (px[i] shr 8) and 0xFF
                else -> px[i] and 0xFF
            }
            buf.put(((v / 255f) - mean[c]) / std[c])
        }
        buf.rewind()

        val input = OnnxTensor.createTensor(env, buf, longArrayOf(1L, 3L, inH.toLong(), inW.toLong()))
        val prob = session.run(mapOf("pixels" to input)).use { out ->
            @Suppress("UNCHECKED_CAST")
            (out[0].value as Array<Array<Array<FloatArray>>>)[0][1]   // channel 1 = polyp
        }

        val scale = minOf(1f, 1024f / src.width)
        val w = (src.width * scale).toInt(); val h = (src.height * scale).toInt()
        val outBmp = Bitmap.createScaledBitmap(src, w, h, true).copy(Bitmap.Config.ARGB_8888, true)
        val opx = IntArray(w * h)
        outBmp.getPixels(opx, 0, w, 0, 0, w, h)

        var hit = 0; var minX = w; var minY = h; var maxX = -1; var maxY = -1
        var sumP = 0f; var peak = 0f; var sumX = 0L; var sumY = 0L
        for (y in 0 until h) {
            val my = y * inH / h
            for (x in 0 until w) {
                if (exclude126 != null &&
                    exclude126[(y * 126 / h) * 126 + (x * 126 / w)]) continue
                val pv = prob[my][x * inW / w]
                if (pv > 0.5f) {
                    hit++
                    if (x < minX) minX = x; if (x > maxX) maxX = x
                    if (y < minY) minY = y; if (y > maxY) maxY = y
                    sumX += x; sumY += y
                    val p = opx[y * w + x]
                    sumP += pv; if (pv > peak) peak = pv
                    val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
                    opx[y * w + x] = (0xFF shl 24) or ((r * 7 / 10) shl 16) or
                            (minOf(255, g * 7 / 10 + 90) shl 8) or (b * 7 / 10)
                }
            }
        }
        outBmp.setPixels(opx, 0, w, 0, 0, w, h)

        val areaPct = 100f * hit / (w * h)
        val found = areaPct > 0.1f
        if (found) {
            val cv = Canvas(outBmp)
            val box = Paint().apply {
                color = Color.rgb(90, 255, 90); style = Paint.Style.STROKE; strokeWidth = 4f }
            cv.drawRect(minX.toFloat(), minY.toFloat(), maxX.toFloat(), maxY.toFloat(), box)
            val tp = Paint().apply {
                color = Color.rgb(90, 255, 90); textSize = 34f; isFakeBoldText = true }
            cv.drawText("polyp", minX.toFloat(), maxOf(36f, minY - 10f), tp)
        }
        var grid: BooleanArray? = null
        if (gridN > 0) {
            grid = BooleanArray(gridN * gridN)
            for (gy in 0 until gridN) for (gx in 0 until gridN) {
                if (exclude126 != null &&
                    exclude126[(gy * 126 / gridN) * 126 + (gx * 126 / gridN)]) continue
                grid[gy * gridN + gx] = prob[gy * inH / gridN][gx * inW / gridN] > 0.5f
            }
        }
        return Result(outBmp, found, areaPct, if (hit > 0) sumP / hit else 0f, peak,
            maxOf(0, maxX - minX), maxOf(0, maxY - minY),
            if (hit > 0) sumX.toFloat() / hit / w else 0.5f,
            if (hit > 0) sumY.toFloat() / hit / h else 0.5f, w, h, grid)
    }
}