package com.example.arguslite

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max

/** Pre-registered Huber port trial, gates P and A (results/huber_port_gate.md).
 *  Runs the exact shipped tracker (Vo) on app-convention fixtures exported by
 *  speqslam/export_tracker_fixtures.py and writes per-arm trajectories for the
 *  Python judge (speqslam/judge_huber_port.py), which owns the ATE metric.
 *  This is a local validation rig: it skips when the fixture directory is absent. */
class TrackerAbTest {
    private val fixtures = File("C:/Users/moonw/EndoscopicMonocularSLAM/data/trackerab_fixtures")

    private fun readFloats(f: File): FloatArray {
        val bytes = f.readBytes()
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    @Test
    fun gatePParityAndGateATrajectories() {
        assumeTrue(fixtures.isDirectory)

        // Gate P: the Kotlin weight function must reproduce the Python reference
        val res = readFloats(File(fixtures, "residuals.bin"))
        val gold = readFloats(File(fixtures, "weights.bin"))
        val wk = Vo.medianScaledWeights(res, res.size)
        var mx = 0f
        for (i in res.indices) mx = max(mx, abs(wk[i] - gold[i]))
        println("GATE_P max |w_kotlin - w_python| = $mx")
        assertTrue("Gate P parity failed: $mx", mx < 1e-5f)

        // Gate A inputs: both arms over each fixture window, trajectories out
        val n = Vo.W * Vo.W
        val outDir = File("build/trackerab").apply { mkdirs() }
        val mask = BooleanArray(n)                 // all-false, per the pre-registration
        for (seq in listOf("c1_ascending_t4_v1_s1", "c1_sigmoid1_t1_v1_s1")) {
            val gray = readFloats(File(File(fixtures, seq), "gray.bin"))
            val depth = readFloats(File(File(fixtures, seq), "depth.bin"))
            val frames = gray.size / n
            for (m in Vo.Mode.values()) {
                Vo.resetStats()
                var pose = StudyEngine.Pose.identity()
                val sb = StringBuilder("0,0,0\n")
                for (k in 1 until frames) {
                    val gA = gray.copyOfRange((k - 1) * n, k * n)
                    val dA = depth.copyOfRange((k - 1) * n, k * n)
                    val gB = gray.copyOfRange(k * n, (k + 1) * n)
                    val rel = Vo.trackPair(gA, dA, gB, mask, m)
                    pose = pose.compose(rel.inverse())   // StudyEngine's composition
                    sb.append("${pose.t[0]},${pose.t[1]},${pose.t[2]}\n")
                }
                File(outDir, "${seq}_${m.name}.csv").writeText(sb.toString())
                println("GATE_A $seq ${m.name}: ${frames - 1} pairs, " +
                        "${Vo.gnBails} bails of ${Vo.gnSolves} solves, " +
                        "avg ${Vo.gnPixels / maxOf(1, Vo.gnSolves)} px/iter")
            }
        }
    }
}
