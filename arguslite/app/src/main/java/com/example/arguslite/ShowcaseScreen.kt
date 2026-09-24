package com.example.arguslite

import android.content.Context
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

/** The whole project in ~45 hands-free seconds: the validated 3D surface, coverage,
 *  blind spots, the real-footage run with its finding, and the numbers. Tap to skip. */

private class HkMap(val n: Int, val xyz: FloatArray, val r: IntArray, val g: IntArray,
                    val b: IntArray, val born: IntArray, val trajN: Int,
                    val traj: FloatArray, val lesions: List<FloatArray>)

private fun loadHk(context: Context): HkMap? = runCatching {
    val bb = java.nio.ByteBuffer.wrap(context.assets.open("hk_map.bin").readBytes())
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
    val n = bb.int
    val xyz = FloatArray(3 * n)
    val r = IntArray(n); val g = IntArray(n); val b = IntArray(n)
    val born = IntArray(n)
    for (i in 0 until n) {
        xyz[3 * i] = bb.float; xyz[3 * i + 1] = bb.float; xyz[3 * i + 2] = bb.float
        r[i] = bb.get().toInt() and 0xFF
        g[i] = bb.get().toInt() and 0xFF
        b[i] = bb.get().toInt() and 0xFF
        bb.get()
        born[i] = bb.short.toInt() and 0xFFFF
    }
    val m = bb.int
    val traj = FloatArray(3 * m)
    for (i in 0 until 3 * m) traj[i] = bb.float
    val lesions = mutableListOf<FloatArray>()
    if (bb.remaining() >= 4) {
        val nl = bb.int
        repeat(nl) { lesions.add(floatArrayOf(bb.float, bb.float, bb.float, bb.float)) }
    }
    HkMap(n, xyz, r, g, b, born, m, traj, lesions)
}.getOrNull()

@Composable
fun ShowcaseScreen() {
    val context = LocalContext.current
    val refMesh = remember { runCatching {
        loadMeshBin(context.assets.open("mesh_cecum.bin").readBytes()) }.getOrNull() }
    val refCS = remember(refMesh) { refMesh?.let { meshCenterSpan(it) } }
    val hk = remember { loadHk(context) }
    var stage by remember { mutableIntStateOf(0) }
    var yaw by remember { mutableFloatStateOf(0.4f) }
    var replayT by remember { mutableFloatStateOf(0f) }
    val durations = listOf(8000L, 7000L, 7000L, 12000L, 6000L, 9000L)

    LaunchedEffect(stage) {
        if (stage == 3) replayT = 0f
        delay(durations[stage])
        stage = (stage + 1) % 6
    }
    LaunchedEffect(Unit) {
        while (true) { delay(50); yaw += 0.006f }
    }
    LaunchedEffect(stage == 3) {
        while (stage == 3) {
            delay(70)
            hk?.let { if (replayT < it.trajN - 1) replayT += 1.2f }
        }
    }
    val pulse by rememberInfiniteTransitionCompat()

    val titles = listOf(
        "A colon, mapped from video alone" to
                "Built by software from a standard endoscope's footage. No added hardware.",
        "Every patch keeps count" to
                "Warm areas were inspected many times. Cold areas were barely glimpsed.",
        "The blind spots" to
                "Red is wall the camera never properly saw. Nobody in the room knows these " +
                "exist. This system does.",
        "Real footage, mapped live" to
                "An actual colonoscopy video, tracked and fused frame by frame.",
        "A polyp: found, placed, sized" to
                "Detection thresholds were fixed in advance on clean footage. Then this " +
                "video ran, and the system found what was there.",
        "Measured, not promised" to "")

    Box(Modifier.fillMaxSize().pointerInput(Unit) {
        detectTapGestures { stage = (stage + 1) % 6 }
    }) {
        if (stage < 5) Canvas(Modifier.fillMaxSize()) {
            when {
                stage <= 2 && refMesh != null && refCS != null ->
                    drawMeshLit(refMesh, refCS[0], refCS[1], refCS[2], refCS[3],
                        yaw, 0.25f, 1.15f, stage, 3)
                stage >= 3 && hk != null -> {
                    val cy = kotlin.math.cos(yaw * 0.4f); val sy = kotlin.math.sin(yaw * 0.4f)
                    val cp = kotlin.math.cos(0.3f); val sp = kotlin.math.sin(0.3f)
                    val s = minOf(size.width, size.height) * 0.40f
                    fun proj(x: Float, y: Float, z: Float): Triple<Float, Float, Float> {
                        val x1 = x * cy + z * sy; val z1 = -x * sy + z * cy
                        val y1 = y * cp - z1 * sp; val z2 = y * sp + z1 * cp
                        val persp = 1.1f / (1.6f - z2 * 0.5f)
                        return Triple(size.width / 2f + x1 * s * persp,
                            size.height / 2f + y1 * s * persp, persp)
                    }
                    val lim = if (stage == 3) replayT else hk.trajN.toFloat()
                    for (i in 0 until hk.n) {
                        if (hk.born[i] > lim) continue
                        val (px, py, persp) = proj(hk.xyz[3*i], hk.xyz[3*i+1], hk.xyz[3*i+2])
                        val shade = (0.5f + 0.6f * persp).coerceIn(0.4f, 1f)
                        drawCircle(Color(hk.r[i] / 255f * shade, hk.g[i] / 255f * shade,
                            hk.b[i] / 255f * shade), 3.4f * persp, Offset(px, py))
                    }
                    var prev: Offset? = null
                    for (k in 0 until minOf(lim.toInt(), hk.trajN)) {
                        val (px, py, _) = proj(hk.traj[3*k], hk.traj[3*k+1], hk.traj[3*k+2])
                        val cur = Offset(px, py)
                        prev?.let { drawLine(Color(0xFF57C7D4), it, cur, 3f) }
                        prev = cur
                    }
                    if (stage == 4 || lim > 10f) hk.lesions.forEach { l ->
                        val (px, py, _) = proj(l[0], l[1], l[2])
                        if (stage == 4)
                            drawCircle(Color(0xFFDCA645), 16f + 10f * pulse,
                                Offset(px, py), alpha = 0.35f)
                        drawCircle(Color(0xFFDCA645), 12f, Offset(px, py))
                        drawCircle(Color(0xFF121519), 5f, Offset(px, py))
                    }
                }
                else -> drawCircle(Color(0xFF2C7B74), 8f,
                    Offset(size.width / 2f, size.height / 2f))
            }
        }
        if (stage == 5) {
            Column(Modifier.align(Alignment.Center).padding(28.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                listOf("0.91 mm" to "camera tracking error vs robot-recorded truth",
                    "98.6 %" to "blind-spot agreement (shown beside its 99.0% do-nothing baseline)",
                    "1.3–2.4 mm" to "3D surface accuracy, from video alone",
                    "11 of 21" to "unseen polyp videos flagged · 17 of 20 controls clean",
                    "9×" to "measured cost of a deforming colon, vs its rigid twin",
                    "2 → 9 fps" to "where whole-colon tracking fails vs holds — the speed limit, measured"
                ).forEach { (n2, d) ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(n2, style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.width(120.dp))
                        Text(d, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
        Column(Modifier.align(Alignment.BottomCenter).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally) {
            Text(titles[stage].first, style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center)
            if (titles[stage].second.isNotEmpty())
                Text(titles[stage].second, style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("tap to continue · ${stage + 1}/6",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline,
                modifier = Modifier.padding(top = 6.dp))
        }
    }
}

@Composable
private fun rememberInfiniteTransitionCompat(): State<Float> {
    val t = remember { mutableFloatStateOf(0f) }
    LaunchedEffect(Unit) {
        var up = true
        while (true) {
            delay(60)
            t.floatValue += if (up) 0.12f else -0.12f
            if (t.floatValue >= 1f) up = false
            if (t.floatValue <= 0f) up = true
        }
    }
    return t
}
