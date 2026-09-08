package com.example.arguslite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import androidx.activity.ComponentActivity

import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.arguslite.ui.theme.ARGUSLiteTheme
import java.util.concurrent.Executors
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableFloatStateOf
import kotlin.math.cos
import kotlin.math.sin
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.detectTapGestures
import kotlinx.coroutines.delay
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.style.TextAlign
import kotlin.time.Duration.Companion.milliseconds

class MainActivity : ComponentActivity() {

    private var heat by mutableStateOf<Bitmap?>(null)
    private var detectStats by mutableStateOf<PolypEngine.Result?>(null)
    private var camCtl by mutableStateOf<LifecycleCameraController?>(null)
    private var depthEngine: DepthEngine? = null
    private var camGranted by mutableStateOf(false)
    private val worker = Executors.newSingleThreadExecutor()
    private var detectImg by mutableStateOf<Bitmap?>(null)
    private var detectMsg by mutableStateOf("Tap a sample case, or pick a photo.")
    private var polypEngine: PolypEngine? = null
    private var recOn by mutableStateOf(false)
    private var recDir: java.io.File? = null
    private var recCount by mutableStateOf(0)

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> camGranted = granted }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        camGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        if (!camGranted) askCamera.launch(Manifest.permission.CAMERA)
        setContent { ARGUSLiteTheme { ArgusApp() } }
    }

    private fun startCamera() {
        val engine = depthEngine ?: DepthEngine(this).also { depthEngine = it }
        val c = LifecycleCameraController(this)
        c.setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
        c.setImageAnalysisAnalyzer(worker) { proxy ->
            val bmp = proxy.toBitmap()
            val m = Matrix().apply { postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
            val upright = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            heat = engine.run(upright)
            if (recOn) recDir?.let { d ->
                if (recCount < 400) {
                    java.io.FileOutputStream(java.io.File(d, "raw%04d.jpg".format(recCount))).use {
                        upright.compress(Bitmap.CompressFormat.JPEG, 80, it) }
                    heat?.let { hb ->
                        java.io.FileOutputStream(java.io.File(d, "heat%04d.jpg".format(recCount))).use {
                            hb.compress(Bitmap.CompressFormat.JPEG, 80, it) } }
                    recCount++
                }
            }
            proxy.close()
        }
        c.bindToLifecycle(this)
        camCtl = c
    }

    @Composable
    fun ArgusApp() {
        var tab by remember { mutableIntStateOf(0) }
        val tabs = listOf(
            "Home" to Icons.Filled.Home,
            "Scope" to Icons.Filled.PlayArrow,
            "Detect" to Icons.Filled.Search,
            "Mission" to Icons.Filled.Info,
            "Results" to Icons.Filled.Star,
            "Map" to Icons.Filled.Place,
        )
        Scaffold(bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { i, (name, icon) ->
                    NavigationBarItem(selected = tab == i, onClick = { tab = i },
                        icon = { Icon(icon, name) }, label = { Text(name) })
                }
            }
        }) { pad ->
            Box(Modifier.padding(pad).fillMaxSize().background(
                androidx.compose.ui.graphics.Brush.verticalGradient(listOf(
                    Color(0xFF121519), Color(0xFF14201E), Color(0xFF121519))))) {
                when (tab) {
                    0 -> HomeScreen { tab = it }
                    1 -> ScopeScreen()
                    2 -> DetectScreen()
                    3 -> MissionScreen()
                    4 -> ResultsScreen()
                    else -> MapScreen()
                }
            }
        }
    }
    private fun analyze(bmp: Bitmap) {
        detectMsg = "analyzing…"
        worker.execute {
            val eng = polypEngine ?: PolypEngine(this).also { polypEngine = it }
            val r = eng.run(bmp)
            detectStats = r
            detectImg = r.overlay
            detectMsg = if (r.found) "polyp detected — covers %.1f%% of the image".format(r.areaPct)
            else "no polyp detected"
        }
    }

    @Composable
    fun DetectScreen() {
        val thumbs = remember {
            (1..6).map { n -> BitmapFactory.decodeStream(assets.open("samples/sample$n.jpg")) }
        }
        val picker = rememberLauncherForActivityResult(
            ActivityResultContracts.PickVisualMedia()) { uri ->
            uri?.let {
                val bmp = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val src = ImageDecoder.createSource(contentResolver, it)
                    ImageDecoder.decodeBitmap(src) { d, _, _ ->
                        d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    }
                } else {
                    contentResolver.openInputStream(it).use { stream ->
                        BitmapFactory.decodeStream(stream)
                    }
                }
                bmp?.let { b -> analyze(b) }
            }
        }
        Column(Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Polyp finder", style = MaterialTheme.typography.headlineMedium)
            Text("These are real colonoscopy photographs. Tap one and a network trained on " +
                    "expert annotations outlines any polyp it finds, right on this phone.")
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(thumbs) { t ->
                    Image(t.asImageBitmap(), "sample",
                        Modifier
                            .height(84.dp)
                            .clickable { analyze(t) })
                }
            }
            Button(onClick = {
                picker.launch(PickVisualMediaRequest(
                    ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) { Text("Pick from gallery") }
            Text(detectMsg, style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary)
            var statsOpen by remember { mutableStateOf(false) }
            var shown by remember { mutableStateOf(0) }
            LaunchedEffect(statsOpen) {
                shown = 0
                if (statsOpen) repeat(4) { delay(160); shown++ }
            }
            detectImg?.let {
                Image(it.asImageBitmap(), "result",
                    Modifier.fillMaxWidth().clickable { statsOpen = !statsOpen })
            }
            detectStats?.let { s ->
                if (s.found && !statsOpen)
                    Text("tap the image for the numbers",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (s.found && statsOpen) {
                    val rows = listOf(
                        "average confidence" to "%.0f%%".format(s.avgConf * 100),
                        "peak confidence" to "%.0f%%".format(s.peakConf * 100),
                        "share of image" to "%.1f%%".format(s.areaPct),
                        "outline size" to "${s.boxW} × ${s.boxH} px")
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        rows.forEachIndexed { i, (k, v) ->
                            androidx.compose.animation.AnimatedVisibility(shown > i) {
                                Card(Modifier.fillMaxWidth()) {
                                    Row(Modifier.padding(12.dp)) {
                                        Text(k, Modifier.weight(1f),
                                            style = MaterialTheme.typography.bodyMedium)
                                        Text(v, style = MaterialTheme.typography.labelLarge,
                                            color = MaterialTheme.colorScheme.primary)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    @Composable
    fun HomeScreen(onGo: (Int) -> Unit) {
        val map = remember { loadColonMap() }
        var spin by remember { mutableFloatStateOf(0f) }
        LaunchedEffect(Unit) { while (true) { spin += 0.006f; delay(16.milliseconds) } }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Canvas(Modifier.fillMaxWidth().height(320.dp)) {
                val cy = cos(spin); val sy = sin(spin)
                val cp = cos(0.35f); val sp = sin(0.35f)
                val s = size.minDimension * 0.40f
                for (i in 0 until map.n step 1) {
                    val x = map.xyz[3 * i]; val y = map.xyz[3 * i + 1]; val z = map.xyz[3 * i + 2]
                    val x1 = x * cy + z * sy; val z1 = -x * sy + z * cy
                    val y1 = y * cp - z1 * sp; val z2 = y * sp + z1 * cp
                    val persp = 1f / (1.6f - z2 * 0.5f)
                    val shade = (0.4f + 0.8f * persp).coerceIn(0.3f, 1f)
                    drawCircle(Color(map.r[i] / 255f * shade, map.g[i] / 255f * shade,
                        map.b[i] / 255f * shade), 2.5f,
                        Offset(size.width / 2 + x1 * s * persp, size.height / 2 + y1 * s * persp))
                }
            }

            val taglines = listOf(
                "the all-seeing monster" to Color(0xFF5FB5AD),

                "leaves no surface unmapped" to Color(0xFFE0796B),
                "one camera. one vision." to Color(0xFF3FD6FF),
                "98.6% accuracy" to Color(0xFFDCA645))
            var tag by remember { mutableIntStateOf(0) }
            LaunchedEffect(Unit) { while (true) { delay(1400.milliseconds); tag = (tag + 1) % taglines.size } }
            androidx.compose.animation.Crossfade(targetState = tag, label = "tag") { i ->
                Text(taglines[i].first, style = MaterialTheme.typography.titleLarge,
                    color = taglines[i].second)
            }
            Text("ARGUS", style = MaterialTheme.typography.displaySmall)
            Text("During a colonoscopy, about one polyp in four slips by unseen. Most of them " +
                    "hide in places the camera never actually looked, and no one knows which places " +
                    "those were. Argus watches the same video the doctor sees and quietly keeps a " +
                    "map of what has been inspected and what has not. This app carries the heart of " +
                    "that system in your pocket.", textAlign = TextAlign.Center)
            val cards = listOf(
                Triple(1, "Scope", "Point the camera at anything and watch a neural network " +
                        "judge its depth, live on this phone."),
                Triple(2, "Detect", "Real colonoscopy cases, analyzed on this phone. Find the polyp."),
                Triple(3, "Mission", "What this project set out to prove, and how it went."),
                Triple(4, "Results", "The numbers, measured against ground truth."),
                Triple(5, "Map", "A real colon in 3D. Spin it, zoom it, and watch it build itself."))
            cards.forEach { (idx, title, desc) ->
                Card(Modifier.fillMaxWidth().clickable { onGo(idx) }) {
                    Column(Modifier.padding(16.dp)) {
                        Text(title, style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.primary)
                        Text(desc, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }

    @Composable
    fun ScopeScreen() {
        var relief by remember { mutableStateOf(false) }
        var rYaw by remember { mutableFloatStateOf(0.5f) }
        var rPitch by remember { mutableFloatStateOf(2.6f) }
        var backLens by remember { mutableStateOf(true) }
        val ctl = camCtl
        Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Scope", style = MaterialTheme.typography.headlineMedium)
            if (ctl == null) {
                Text("This is stage one of the Argus pipeline, running live. The same network " +
                        "that maps colons will look through your camera and judge how far away " +
                        "everything is, using nothing but a single flat image at a time.")
                Button(onClick = {
                    if (camGranted) startCamera()
                    else askCamera.launch(Manifest.permission.CAMERA)
                }) { Text("Start the camera") }
                return@Column
            }
            AndroidView(
                factory = { ctx -> PreviewView(ctx).also { it.controller = ctl } },
                modifier = Modifier.fillMaxWidth().weight(1f)
            )
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    backLens = !backLens
                    ctl.cameraSelector = if (backLens) CameraSelector.DEFAULT_BACK_CAMERA
                    else CameraSelector.DEFAULT_FRONT_CAMERA
                }) { Text("Flip") }
                Switch(checked = relief, onCheckedChange = { relief = it })
                Text(if (relief) "3D relief" else "heat map")
            }
            @Composable
            fun SessionBar() {
                var showList by remember { mutableStateOf(false) }
                var playDir by remember { mutableStateOf<java.io.File?>(null) }
                val root = java.io.File(filesDir, "sessions").apply { mkdirs() }
                Row(verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(colors = ButtonDefaults.buttonColors(
                        containerColor = if (recOn) Color(0xFFE63946) else MaterialTheme.colorScheme.primary),
                        onClick = {
                            if (recOn) { recOn = false }
                            else {
                                val d = java.io.File(root, "scan " +
                                        java.text.SimpleDateFormat("MMM d HH.mm.ss").format(java.util.Date()))
                                d.mkdirs(); recDir = d; recCount = 0; recOn = true
                            }
                        }) { Text(if (recOn) "Stop  ●  ${recCount}" else "Record") }
                    TextButton(onClick = { showList = !showList }) { Text("Sessions") }
                }
                if (showList) {
                    val sessions = root.listFiles()?.sortedByDescending { it.name } ?: emptyList()
                    if (sessions.isEmpty()) Text("no saved scans yet",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    sessions.forEach { s ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(onClick = { playDir = s }) { Text(s.name) }
                            Spacer(Modifier.weight(1f))
                            TextButton(onClick = { s.deleteRecursively(); showList = false; showList = true })
                            { Text("delete", color = MaterialTheme.colorScheme.secondary) }
                        }
                    }
                }
                playDir?.let { d ->
                    val n = (d.listFiles()?.count { it.name.startsWith("raw") } ?: 0)
                    if (n == 0) { playDir = null; return }
                    var i by remember(d) { mutableStateOf(0) }
                    LaunchedEffect(d) { while (true) { delay(250); i = (i + 1) % n } }
                    val raw = remember(i) { BitmapFactory.decodeFile(java.io.File(d, "raw%04d.jpg".format(i)).path) }
                    val ht = remember(i) { BitmapFactory.decodeFile(java.io.File(d, "heat%04d.jpg".format(i)).path) }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        raw?.let { Image(it.asImageBitmap(), null, Modifier.weight(1f)) }
                        ht?.let { Image(it.asImageBitmap(), null, Modifier.weight(1f)) }
                    }
                    Row {
                        Text("${d.name}  ·  frame ${i + 1}/$n", Modifier.weight(1f),
                            style = MaterialTheme.typography.bodySmall)
                        TextButton(onClick = { playDir = null }) { Text("close") }
                    }
                }
            }
            if (!relief) {
                heat?.let { Image(it.asImageBitmap(), "depth", Modifier.fillMaxWidth().weight(1f)) }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("far ", style = MaterialTheme.typography.bodySmall)
                    Canvas(Modifier.weight(1f).height(10.dp)) {
                        drawRect(androidx.compose.ui.graphics.Brush.horizontalGradient(
                            listOf(Color(0xFF1A0033), Color(0xFFC03030), Color(0xFFFFE060))))
                    }
                    Text(" near", style = MaterialTheme.typography.bodySmall)
                }
                SessionBar()
            } else {
                val h = heat  // reading this makes the relief redraw with every new frame
                val grid = depthEngine?.lastGrid
                val gn = depthEngine?.gridN ?: 42
                Canvas(Modifier.fillMaxWidth().weight(1f).pointerInput(Unit) {
                    detectDragGestures { _, d -> rYaw += d.x * 0.008f; rPitch += d.y * 0.008f }
                }) {
                    if (grid == null || h == null) return@Canvas
                    val cy = cos(rYaw); val sy = sin(rYaw)
                    val cp = cos(rPitch); val sp = sin(rPitch)
                    val s = size.minDimension * 0.55f
                    for (gy in 0 until gn) for (gx in 0 until gn) {
                        val t = grid[gy * gn + gx]
                        val x = gx / (gn - 1f) - 0.5f
                        val y = gy / (gn - 1f) - 0.5f
                        val z = (t - 0.5f) * 0.7f
                        val x1 = x * cy + z * sy; val z1 = -x * sy + z * cy
                        val y1 = y * cp - z1 * sp; val z2 = y * sp + z1 * cp
                        val persp = 1f / (1.7f - z2 * 0.6f)
                        drawCircle(Color(minOf(1f, t * 2f), maxOf(0f, t * 2f - 1f),
                            maxOf(0f, 0.35f - t * 0.5f) + 0.05f), 3.5f,
                            Offset(size.width / 2 + x1 * s * persp,
                                size.height / 2 + y1 * s * persp))
                    }
                }
                Text("every dot is a pixel, lifted off the screen by its predicted depth — " +
                        "drag to look around it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }

    @Composable
    fun ExpandCard(title: String, body: String) {
        var open by remember { mutableStateOf(false) }
        Card(Modifier.fillMaxWidth().clickable { open = !open }) {
            Column(Modifier.padding(16.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary)
                androidx.compose.animation.AnimatedVisibility(open) {
                    Text(body, Modifier.padding(top = 8.dp),
                        style = MaterialTheme.typography.bodyMedium)
                }
            }
        }
    }

    @Composable
    fun MissionScreen() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("The mission", style = MaterialTheme.typography.headlineMedium)
            Text("Tap any section to read it.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
            ExpandCard("What a polyp is",
                "A polyp is a small growth on the inner wall of the colon. Most are harmless, " +
                        "but some slowly turn into colorectal cancer over the years. That is why " +
                        "screening colonoscopies exist. If you find a polyp early and remove it, the " +
                        "cancer never happens.")
            ExpandCard("The problem nobody could see",
                "Studies keep finding that roughly one polyp in four gets missed during " +
                        "colonoscopy. The main reason is simple and a little unsettling. The camera " +
                        "shows one small window of a long folded tunnel, and some patches of wall " +
                        "simply never appear in that window. The doctor cannot re-check a spot they " +
                        "never knew they skipped.")
            ExpandCard("The question this project asked",
                "Can software figure out which parts of the colon were never actually " +
                        "inspected, using nothing but the video from a standard endoscope? No new " +
                        "sensors, no hardware changes, just the same feed the doctor already watches.")
            ExpandCard("How Argus answers it",
                "Four stages run in a row. A neural network estimates how far away every pixel " +
                        "is. A tracking algorithm works out how the camera moved between frames. A " +
                        "fusion step melts thousands of those noisy snapshots into one clean 3D " +
                        "surface. Finally, every patch of that surface is checked. If fewer than three " +
                        "frames ever saw a patch properly, it gets flagged as a blind spot.")
            ExpandCard("What went wrong on the way",
                "The first idea was that lens glare was breaking everything. Careful testing " +
                        "proved that mostly wrong. Later, the whole system collapsed when it switched " +
                        "to guessing depth with a generic AI model. A controlled experiment traced the " +
                        "failure to that one component. The model had learned the outside world and " +
                        "had never seen the inside of a body.")
            ExpandCard("What we found",
                "After teaching the depth network what colons actually look like, the system " +
                        "went from 74 percent agreement with ground truth to 98.6 percent. The finding " +
                        "is bigger than the gadget. Generic AI vision fails inside the human body until " +
                        "you train it on the territory, and one targeted fix in the right place can " +
                        "rescue an entire system.")
        }
    }

    @Composable
    fun ResultsScreen() {
        var speed by remember { mutableFloatStateOf(1f) }
        val seqs = listOf(
            Triple("cecum", Color(0xFF5FB5AD), floatArrayOf(0.8f, 1.6f, 2.6f, 3.8f, 5.6f)),
            Triple("descending", Color(0xFFE0796B), floatArrayOf(2.9f, 6.0f, 8.3f, 10.7f, 14.9f)),
            Triple("transverse 1", Color(0xFF3FD6FF), floatArrayOf(1.9f, 2.7f, 3.4f, 4.6f, 6.4f)),
            Triple("transverse 2a", Color(0xFFDCA645), floatArrayOf(0.1f, 0.2f, 0.4f, 0.5f, 5.8f)),
            Triple("transverse 2b", Color(0xFFA29A90), floatArrayOf(0.4f, 0.6f, 0.9f, 1.6f, 2.3f)))
        val xs = floatArrayOf(1f, 2f, 3f, 4f, 6f)
        fun interp(ys: FloatArray, s: Float): Float {
            var i = 0
            while (i < 3 && xs[i + 1] < s) i++
            val f = (s - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + (ys[i + 1] - ys[i]) * f.coerceIn(0f, 1f)
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Results", style = MaterialTheme.typography.headlineMedium)
            Text("Every number here was measured against ground truth on a phantom whose real " +
                    "shape and camera path were recorded by machines.", textAlign = TextAlign.Center)
            Card {
                Column(Modifier.padding(16.dp)) {
                    Text("The cost of rushing", style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary)
                    Text("Drag the slider. You are speeding up the scope, and the curves show " +
                            "how much of the wall goes uninspected in five real sequences.",
                        style = MaterialTheme.typography.bodySmall)
                    Canvas(Modifier.fillMaxWidth().height(190.dp).padding(vertical = 8.dp)) {
                        val x0 = 30f; val yMax = 16f
                        fun px(s: Float) = x0 + (s - 1f) / 5f * (size.width - x0 - 10f)
                        fun py(b: Float) = size.height - 20f - b / yMax * (size.height - 30f)
                        seqs.forEach { (_, col, ys) ->
                            for (j in 1 until xs.size) {
                                drawLine(col, Offset(px(xs[j - 1]), py(ys[j - 1])),
                                    Offset(px(xs[j]), py(ys[j])), 4f)
                            }
                            drawCircle(col, 7f, Offset(px(speed), py(interp(ys, speed))))
                        }
                        drawLine(Color(0x66FFFFFF), Offset(px(speed), 0f),
                            Offset(px(speed), size.height - 15f), 2f)
                    }
                    Slider(value = speed, onValueChange = { speed = it }, valueRange = 1f..6f)
                    val vals = seqs.map { interp(it.third, speed) }
                    Text("At ×%.1f speed, %.1f%% to %.1f%% of the wall goes unseen."
                        .format(speed, vals.min(), vals.max()),
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.secondary)
                }
            }
            Card {
                Column(Modifier.padding(16.dp)) {
                    Text("The fix that saved the project",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary)
                    Text("Depth accuracy per sequence, before and after teaching the network " +
                            "what a colon looks like.", style = MaterialTheme.typography.bodySmall)
                    val pairs = listOf(64f to 98.9f, 69f to 91.8f, 66f to 97.9f,
                        74f to 99.7f, 67f to 99.6f)
                    Canvas(Modifier.fillMaxWidth().height(150.dp).padding(top = 8.dp)) {
                        val bw = size.width / (pairs.size * 3f)
                        pairs.forEachIndexed { i, (a, b) ->
                            val cx = size.width * (i + 0.5f) / pairs.size
                            drawRect(Color(0xFFE0796B),
                                Offset(cx - bw, size.height * (1 - a / 100f)),
                                androidx.compose.ui.geometry.Size(bw * 0.9f, size.height * a / 100f))
                            drawRect(Color(0xFF5FB5AD),
                                Offset(cx, size.height * (1 - b / 100f)),
                                androidx.compose.ui.geometry.Size(bw * 0.9f, size.height * b / 100f))
                        }
                    }
                    Text("coral is the generic model, teal is after fine tuning",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            listOf("0.91 mm" to "average camera tracking error, using video alone",
                "98.6 %" to "blind spot agreement with ground truth",
                "1.3 to 2.4 mm" to "3D surface accuracy, from video alone",
                "0.60 mm" to "median measurement error on polyp sized targets").forEach { (n, d) ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp)) {
                        Text(n, style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary)
                        Text(d, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }

    class ColonMap(val n: Int, val xyz: FloatArray, val r: IntArray, val g: IntArray,
                   val b: IntArray, val blind: BooleanArray, val born: IntArray,
                   val trajN: Int, val traj: FloatArray)

    class Anchor(val x: Float, val y: Float, val z: Float, val title: String, val body: String)

    private fun loadColonMap(): ColonMap {
        val bb = java.nio.ByteBuffer.wrap(assets.open("colon_map2.bin").readBytes())
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val n = bb.int
        val xyz = FloatArray(3 * n)
        val r = IntArray(n); val g = IntArray(n); val b = IntArray(n)
        val blind = BooleanArray(n); val born = IntArray(n)
        for (i in 0 until n) {
            xyz[3 * i] = bb.float; xyz[3 * i + 1] = bb.float; xyz[3 * i + 2] = bb.float
            r[i] = bb.get().toInt() and 0xFF
            g[i] = bb.get().toInt() and 0xFF
            b[i] = bb.get().toInt() and 0xFF
            blind[i] = bb.get().toInt() == 1
            born[i] = bb.short.toInt() and 0xFFFF
        }
        val m = bb.int
        val traj = FloatArray(3 * m)
        for (i in 0 until 3 * m) traj[i] = bb.float
        return ColonMap(n, xyz, r, g, b, blind, born, m, traj)
    }

    private fun makeAnchors(map: ColonMap): List<Anchor> {
        val bi = map.blind.indexOfFirst { it }.coerceAtLeast(0)
        var si = 0; var best = Float.MAX_VALUE
        for (i in 0 until map.n) {
            val dx = map.xyz[3 * i] - map.traj[0]
            val dy = map.xyz[3 * i + 1] - map.traj[1]
            val dz = map.xyz[3 * i + 2] - map.traj[2]
            val d = dx * dx + dy * dy + dz * dz
            if (d < best) { best = d; si = i }
        }
        var di = 0
        for (i in 0 until map.n) if (map.xyz[3 * i + 2] > map.xyz[3 * di + 2]) di = i
        return listOf(
            Anchor(map.xyz[3 * bi], map.xyz[3 * bi + 1], map.xyz[3 * bi + 2], "A blind spot",
                "The camera never got a proper look at this patch. Fewer than three frames saw " +
                        "it. In a real procedure, this is exactly where a polyp could hide unnoticed."),
            Anchor(map.xyz[3 * si], map.xyz[3 * si + 1], map.xyz[3 * si + 2], "Where the run began",
                "The scope entered here. Every point in this model was placed by combining two " +
                        "guesses: how far away each pixel was, and how the camera was moving."),
            Anchor(map.xyz[3 * di], map.xyz[3 * di + 1], map.xyz[3 * di + 2], "The far wall",
                "The depth network judged these distances from single flat images. That is the " +
                        "same trick your brain does when you close one eye."))
    }

    @Composable
    fun MapScreen() {
        val map = remember { loadColonMap() }
        val anchors = remember { makeAnchors(map) }
        var yaw by remember { mutableFloatStateOf(0.6f) }
        var pitch by remember { mutableFloatStateOf(0.3f) }
        var zoom by remember { mutableFloatStateOf(1f) }
        var showBlind by remember { mutableStateOf(false) }
        var replay by remember { mutableStateOf(false) }
        var playing by remember { mutableStateOf(false) }
        var t by remember { mutableFloatStateOf(0f) }
        var picked by remember { mutableStateOf<Anchor?>(null) }
        var canvasW by remember { mutableFloatStateOf(1f) }
        var canvasH by remember { mutableFloatStateOf(1f) }

        fun project(x: Float, y: Float, z: Float, cy: Float, sy: Float, cp: Float, sp: Float): Triple<Float, Float, Float> {
            val x1 = x * cy + z * sy; val z1 = -x * sy + z * cy
            val y1 = y * cp - z1 * sp; val z2 = y * sp + z1 * cp
            val persp = zoom / (1.6f - z2 * 0.5f)
            val s = minOf(canvasW, canvasH) * 0.42f
            return Triple(canvasW / 2f + x1 * s * persp, canvasH / 2f + y1 * s * persp, persp)
        }

        LaunchedEffect(playing) {
            while (playing && t < map.trajN - 1) { t += 2f; delay(70.milliseconds) }
            if (t >= map.trajN - 1) playing = false
        }

        Column(Modifier.fillMaxSize().padding(12.dp)) {
            Text("The 3D map", style = MaterialTheme.typography.headlineMedium)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = showBlind, onCheckedChange = { showBlind = it })
                Text(" blind spots", Modifier.weight(1f))
                Button(onClick = {
                    replay = !replay
                    if (replay) { t = 0f; playing = true } else playing = false
                }) { Text(if (replay) "Explore" else "Watch it build") }
            }
            if (replay) {
                val frameIdx = (t / map.trajN * 91).toInt().coerceIn(0, 91)
                val raw = remember(frameIdx) {
                    BitmapFactory.decodeStream(assets.open("replay/f%03d.jpg".format(frameIdx))) }
                val dep = remember(frameIdx) {
                    BitmapFactory.decodeStream(assets.open("replay/f%03dd.jpg".format(frameIdx))) }
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Image(raw.asImageBitmap(), "camera", Modifier.weight(1f).height(130.dp))
                    Image(dep.asImageBitmap(), "depth", Modifier.weight(1f).height(130.dp))
                }
                Text("what the scope saw · what the network understood · " +
                        "frame ${t.toInt()} of ${map.trajN}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                androidx.compose.animation.AnimatedVisibility(!playing && t >= map.trajN - 1) {
                    Card {
                        Column(Modifier.padding(14.dp)) {
                            Text("Run complete", style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.primary)
                            Text("276 frames fused into one surface. 94.7 percent of the wall " +
                                    "was properly inspected. The rest is what the red dots mark, " +
                                    "and finding them was the whole point.")
                            TextButton(onClick = { t = 0f; playing = true }) { Text("run it again") }
                        }
                    }
                }
            }
            Box(Modifier.fillMaxWidth().weight(1f)) {
                Canvas(Modifier.fillMaxSize()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, gz, _ ->
                            yaw += pan.x * 0.008f; pitch += pan.y * 0.008f
                            zoom = (zoom * gz).coerceIn(0.5f, 4f)
                        }
                    }
                    .pointerInput(Unit) {
                        detectTapGestures { tap ->
                            val cy = cos(yaw); val sy = sin(yaw); val cp = cos(pitch); val sp = sin(pitch)
                            picked = anchors.firstOrNull {
                                val (px, py, _) = project(it.x, it.y, it.z, cy, sy, cp, sp)
                                val dx = tap.x - px; val dy = tap.y - py
                                dx * dx + dy * dy < 55f * 55f
                            }
                        }
                    }) {
                    canvasW = size.width; canvasH = size.height
                    val cy = cos(yaw); val sy = sin(yaw); val cp = cos(pitch); val sp = sin(pitch)
                    for (i in 0 until map.n) {
                        if (replay && map.born[i] > t) continue
                        val (px, py, persp) = project(
                            map.xyz[3 * i], map.xyz[3 * i + 1], map.xyz[3 * i + 2], cy, sy, cp, sp)
                        val shade = (0.45f + 0.75f * persp / zoom).coerceIn(0.35f, 1f)
                        if (showBlind && map.blind[i]) {
                            drawCircle(Color(0xFFE63946), 5f, Offset(px, py))
                        } else {
                            drawCircle(Color(map.r[i] / 255f * shade, map.g[i] / 255f * shade,
                                map.b[i] / 255f * shade), 3f, Offset(px, py),
                                alpha = if (showBlind) 0.30f else 1f)
                        }
                    }
                    val upTo = (if (replay) t.toInt() else map.trajN - 1).coerceAtMost(map.trajN - 1)
                    for (j in 1..upTo) {
                        val (ax, ay, _) = project(map.traj[3 * (j - 1)],
                            map.traj[3 * (j - 1) + 1], map.traj[3 * (j - 1) + 2], cy, sy, cp, sp)
                        val (bx, by, _) = project(map.traj[3 * j],
                            map.traj[3 * j + 1], map.traj[3 * j + 2], cy, sy, cp, sp)
                        drawLine(Color(0xFF3FD6FF), Offset(ax, ay), Offset(bx, by), 3f)
                    }
                    if (!replay) anchors.forEach {
                        val (px, py, _) = project(it.x, it.y, it.z, cy, sy, cp, sp)
                        drawCircle(Color(0xFF5FB5AD), 10f, Offset(px, py), alpha = 0.9f)
                        drawCircle(Color(0xFF121519), 5f, Offset(px, py))
                    }
                }
                picked?.let { a ->
                    Card(Modifier.align(Alignment.BottomCenter).padding(8.dp)) {
                        Column(Modifier.padding(14.dp)) {
                            Text(a.title, style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.primary)
                            Text(a.body, style = MaterialTheme.typography.bodyMedium)
                            TextButton(onClick = { picked = null }) { Text("close") }
                        }
                    }
                }
            }
            Text("drag to turn · pinch to zoom · tap for info",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}