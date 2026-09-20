package com.example.arguslite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import kotlin.math.cos
import kotlin.math.sin

/** Live-study state that survives tab switches. Video streaming runs on its own thread;
 *  camera mode feeds frames in while the screen is open. */
object StreamRunner {
    var engine by mutableStateOf<StreamEngine?>(null)
    var running by mutableStateOf(false)
    var mode by mutableStateOf("")          // "video" | "camera"
    var status by mutableStateOf("")
    var startedAt = 0L
    private var stopFlag = false

    fun ensureEngine(context: Context): StreamEngine {
        val e = engine ?: StreamEngine(DepthEngine(context.applicationContext),
            PolypEngine(context.applicationContext)).also { engine = it }
        return e
    }

    fun startVideo(context: Context, path: String, label: String) {
        if (running) return
        val e = StreamEngine(DepthEngine(context.applicationContext),
            PolypEngine(context.applicationContext))
        engine = e
        running = true; stopFlag = false; mode = "video"
        status = "streaming $label"
        startedAt = System.currentTimeMillis()
        thread {
            try {
                val mmr = MediaMetadataRetriever()
                mmr.setDataSource(path)
                val durMs = mmr.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
                var t = 0L
                val stepMs = 150L                    // ~7 frames per second of footage
                // Missing duration metadata plus a retriever that returns the LAST frame
                // for any time past the end would loop forever — so also detect the
                // end of the stream by the footage going static.
                var lastSig = 0L
                var sameCount = 0
                val capMs = if (durMs > 0) durMs else 5 * 60 * 1000L
                while (!stopFlag && t < capMs) {
                    val f = mmr.getFrameAtTime(t * 1000L,
                        MediaMetadataRetriever.OPTION_CLOSEST) ?: break
                    var sig = 0L
                    for (yy in 0 until 8) for (xx in 0 until 8)
                        sig = sig * 31 + f.getPixel(xx * f.width / 8, yy * f.height / 8)
                    if (sig == lastSig) {
                        sameCount++
                        if (sameCount >= 3) break     // footage stopped changing: done
                        t += stepMs
                        continue
                    }
                    lastSig = sig; sameCount = 0
                    val w = 512
                    val h = maxOf(1, w * f.height / f.width)
                    val s = Bitmap.createScaledBitmap(f, w, h, true)
                    if (s !== f) f.recycle()
                    e.step(s)
                    t += stepMs
                }
                mmr.release()
                File(path).takeIf { it.name.startsWith("stream_") }?.delete()
            } catch (ex: Exception) {
                status = "stream ended: ${ex.message}"
            }
            finishInternal(context, label)
        }
    }

    fun startCamera(context: Context) {
        if (running) return
        ensureEngine(context)
        running = true; stopFlag = false; mode = "camera"
        status = "live camera study"
        startedAt = System.currentTimeMillis()
    }

    fun stop(context: Context, label: String) {
        stopFlag = true
        if (mode == "camera") finishInternal(context, label)
        // video mode finishes itself when its loop sees the flag
    }

    private fun finishInternal(context: Context, label: String) {
        val e = engine
        if (e != null && e.snapshot().frames > 5) {
            val secs = (System.currentTimeMillis() - startedAt) / 1000f
            StudyRunner.init(context)
            StudyRunner.result = e.finish(secs)
            StudyRunner.sourceName = "$label (live stream)"
            status = "saved — open the Studies tab to replay"
        } else status = "stream too short to save"
        running = false; mode = ""
    }
}

@Composable
fun StreamScreen() {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current
    var link by rememberSaveable { mutableStateOf("") }
    var tick by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) { while (true) { delay(700); tick++ } }
    val live = remember(tick, StreamRunner.engine) { StreamRunner.engine?.snapshot() }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            thread {
                try {
                    val dest = File(context.cacheDir, "stream_pick.mp4")
                    context.contentResolver.openInputStream(it)!!.use { input ->
                        dest.outputStream().use { out -> input.copyTo(out) }
                    }
                    StreamRunner.startVideo(context, dest.path, "picked video")
                } catch (e: Exception) { StreamRunner.status = "could not open video" }
            }
        }
    }

    // camera feed while this screen is open in camera mode
    val camExec = remember { Executors.newSingleThreadExecutor() }
    if (StreamRunner.running && StreamRunner.mode == "camera") {
        DisposableEffect(Unit) {
            val ctl = LifecycleCameraController(context)
            ctl.cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            ctl.setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
            ctl.setImageAnalysisAnalyzer(camExec) { proxy ->
                val e = StreamRunner.engine
                if (e != null && !e.busy) {
                    val bmp = proxy.toBitmap()
                    val m = Matrix().apply {
                        postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
                    val up = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                    e.step(up)
                }
                proxy.close()
            }
            ctl.bindToLifecycle(lifecycle)
            onDispose { ctl.unbind() }
        }
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Live study", style = MaterialTheme.typography.headlineMedium)

        if (!StreamRunner.running && (live == null || live.frames == 0)) {
            Text("The full pipeline, running as the footage plays: the 3D map, the " +
                    "coverage count, and the polyp screen all grow frame by frame. " +
                    "Coverage is counted on every processed frame. Point it at a video, " +
                    "a link, or the camera.")
            Button(onClick = { picker.launch("video/*") }, Modifier.fillMaxWidth()) {
                Text("Stream a video from this phone")
            }
            OutlinedTextField(value = link, onValueChange = { link = it },
                label = { Text("Or a video link") }, singleLine = true,
                modifier = Modifier.fillMaxWidth())
            Button(onClick = {
                val url = link.trim()
                thread {
                    try {
                        val dest = File(context.cacheDir, "stream_fetch.mp4")
                        val conn = URL(url).openConnection() as HttpURLConnection
                        conn.inputStream.use { i -> dest.outputStream().use { o -> i.copyTo(o) } }
                        StreamRunner.startVideo(context, dest.path,
                            url.substringAfterLast('/'))
                    } catch (e: Exception) { StreamRunner.status = "fetch failed" }
                }
            }, enabled = link.isNotBlank(), modifier = Modifier.fillMaxWidth()) {
                Text("Fetch and stream")
            }
            Button(onClick = { StreamRunner.startCamera(context) },
                Modifier.fillMaxWidth()) { Text("Use the camera") }
            if (StreamRunner.status.isNotEmpty())
                Text(StreamRunner.status, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        if (StreamRunner.running && live != null && live.frames == 0) {
            Text("Calibrating: ${live.warmDone} of ${live.warmTotal} frames — learning " +
                    "which pixels are scene and which are overlay before measuring anything.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.primary)
            LinearProgressIndicator(
                progress = { live.warmDone.toFloat() / live.warmTotal },
                Modifier.fillMaxWidth())
        }

        if (live != null && live.frames > 0) {
            var showBlind by remember { mutableStateOf(true) }
            var yaw by remember { mutableFloatStateOf(0.6f) }
            var pitch by remember { mutableFloatStateOf(0.3f) }
            var zoom by remember { mutableFloatStateOf(1f) }
            Canvas(Modifier.fillMaxWidth().height(280.dp)
                .pointerInput(Unit) {
                    detectTransformGestures { _, pan, gz, _ ->
                        yaw += pan.x * 0.008f; pitch += pan.y * 0.008f
                        zoom = (zoom * gz).coerceIn(0.4f, 3f)
                    }
                }) {
                val cyw = cos(yaw); val syw = sin(yaw)
                val cp = cos(pitch); val sp = sin(pitch)
                fun proj(x: Float, y: Float, z: Float): Triple<Float, Float, Float> {
                    val x1 = x * cyw + z * syw; val z1 = -x * syw + z * cyw
                    val y1 = y * cp - z1 * sp; val z2 = y * sp + z1 * cp
                    val persp = zoom / (1.6f - z2 * 0.5f)
                    val s = minOf(size.width, size.height) * 0.42f
                    return Triple(size.width / 2f + x1 * s * persp,
                        size.height / 2f + y1 * s * persp, persp)
                }
                // painter's order + depth shading: far wall first and dimmer, so the
                // growing cloud reads as a tube rather than a blob
                val m = live.pts.size / 3
                val pxA = FloatArray(m); val pyA = FloatArray(m); val ppA = FloatArray(m)
                for (i in 0 until m) {
                    val (px, py, persp) = proj(live.pts[3*i], live.pts[3*i+1], live.pts[3*i+2])
                    pxA[i] = px; pyA[i] = py; ppA[i] = persp
                }
                for (i in (0 until m).sortedBy { ppA[it] }) {
                    val persp = ppA[i]
                    if (persp <= 0f) continue
                    val shade = (0.45f + 0.75f * persp / zoom).coerceIn(0.35f, 1f)
                    val c = if (showBlind && live.blind[i])
                        Color(0.88f * shade, 0.24f * shade, 0.26f * shade)
                    else Color(((live.colors[i] shr 16) and 0xFF) / 255f * shade,
                        ((live.colors[i] shr 8) and 0xFF) / 255f * shade,
                        (live.colors[i] and 0xFF) / 255f * shade)
                    drawCircle(c, 1.7f * persp, Offset(pxA[i], pyA[i]))
                }
                var prev: Offset? = null
                for (k in 0 until live.traj.size / 3) {
                    val (px, py, _) = proj(live.traj[3*k], live.traj[3*k+1], live.traj[3*k+2])
                    val cur = Offset(px, py)
                    prev?.let { drawLine(Color(0xFF57C7D4), it, cur, 3f) }
                    prev = cur
                }
                live.lesions.forEach { l ->
                    val (px, py, persp) = proj(l.x, l.y, l.z)
                    drawCircle(Color(0xFFE8B33C), 9f * persp, Offset(px, py))
                    drawCircle(Color(0xFF121519), 4f * persp, Offset(px, py))
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically) {
                live.thumb?.let {
                    Image(it.asImageBitmap(), null, Modifier.width(100.dp))
                }
                Column(Modifier.weight(1f)) {
                    Text("frame ${live.frames} · ${live.voxels} patches",
                        style = MaterialTheme.typography.labelLarge)
                    Text("under-seen so far: %.1f%% · path %.0f mm"
                        .format(live.blindPct, live.pathMm),
                        style = MaterialTheme.typography.bodySmall)
                    Text("ms/frame — depth %.0f · track %.0f · fuse %.0f · detect %.0f"
                        .format(live.stageMs[0], live.stageMs[1], live.stageMs[2],
                            live.stageMs[3]),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (live.lesions.isNotEmpty())
                        Text("POLYP PINNED — ~%.1f mm".format(live.lesions[0].sizeMm),
                            color = Color(0xFFE8B33C),
                            style = MaterialTheme.typography.labelLarge)
                    else if (live.bestSupport > 0)
                        Text("polyp evidence: ${live.bestSupport} of " +
                                "${live.needSupport} needed at one spot",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color(0xFFE8B33C))
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = showBlind, onCheckedChange = { showBlind = it })
                Text("Highlight under-seen wall (red)", Modifier.weight(1f))
                TextButton(onClick = {
                    if (live.traj.size >= 6) {
                        val li = live.traj.size - 3
                        val dx = live.traj[li] - live.traj[0]
                        val dy = live.traj[li + 1] - live.traj[1]
                        val dz = live.traj[li + 2] - live.traj[2]
                        val ny = kotlin.math.atan2(-dx, dz)
                        val z1 = -dx * sin(ny) + dz * cos(ny)
                        yaw = ny; pitch = kotlin.math.atan2(dy, z1); zoom = 1.3f
                    }
                }) { Text("down the tube") }
            }
            if (StreamRunner.running) {
                Button(onClick = { StreamRunner.stop(context,
                    if (StreamRunner.mode == "camera") "camera" else "video") },
                    Modifier.fillMaxWidth()) { Text("Stop and save to Studies") }
                Text(StreamRunner.status, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (StreamRunner.mode == "camera")
                    Text("Camera streaming runs while this screen is open.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                Text(StreamRunner.status, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Button(onClick = { StreamRunner.engine = null; StreamRunner.status = "" },
                    Modifier.fillMaxWidth()) { Text("New live study") }
            }
            Text("Numbers here are live estimates: uncalibrated camera, streaming " +
                    "statistics. A stopped stream is saved as a study and can be " +
                    "replayed from the Studies tab.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
