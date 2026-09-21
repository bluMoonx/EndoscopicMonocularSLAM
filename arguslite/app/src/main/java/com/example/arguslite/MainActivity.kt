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
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.RepeatMode
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.time.Duration.Companion.milliseconds
import androidx.compose.foundation.layout.aspectRatio
import kotlin.math.sqrt
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString

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
    private var recCount by mutableIntStateOf(0)

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

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun ArgusApp() {
        var tab by remember { mutableIntStateOf(2) }
        var showMore by remember { mutableStateOf(false) }
        var showcase by remember { mutableStateOf(false) }
        val tabs = listOf(
            "Mission" to Icons.Filled.Info,
            "Results" to Icons.Filled.Star,
            "Home" to Icons.Filled.Home,
            "Studies" to Icons.Filled.Place,
            "More" to Icons.Filled.Menu,
        )
        Scaffold(bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { i, (name, icon) ->
                    NavigationBarItem(selected = if (i < 4) tab == i else tab > 3,
                        onClick = { if (i < 4) tab = i else showMore = true },
                        icon = { Icon(icon, name) }, label = { Text(name) })
                }
            }
        }) { pad ->
            Box(Modifier.padding(pad).fillMaxSize().background(
                androidx.compose.ui.graphics.Brush.verticalGradient(listOf(
                    Color(0xFF121519), Color(0xFF14201E), Color(0xFF121519))))) {
                when (tab) {
                    0 -> MissionScreen()
                    1 -> ResultsScreen()
                    2 -> HomeScreen(onGo = { tab = it }, onShowcase = { showcase = true })
                    3 -> MapScreen()
                    4 -> ScopeScreen()
                    5 -> DetectScreen()
                    6 -> MeasureScreen()
                    7 -> TrajScreen()
                    8 -> XRayScreen()
                    9 -> GlareLabScreen()
                    else -> StreamScreen()
                }
                // Showcase is a moment, not a tab: tapping the Home model zooms it in,
                // and the X (or back-tap on the close) zooms it away.
                androidx.compose.animation.AnimatedVisibility(
                    visible = showcase,
                    enter = androidx.compose.animation.fadeIn(tween(450)) +
                            androidx.compose.animation.scaleIn(
                                initialScale = 0.62f, animationSpec = tween(450)),
                    exit = androidx.compose.animation.fadeOut(tween(240)) +
                            androidx.compose.animation.scaleOut(
                                targetScale = 0.80f, animationSpec = tween(240))) {
                    Box(Modifier.fillMaxSize().background(Color(0xFF121519))) {
                        ShowcaseScreen()
                        IconButton(onClick = { showcase = false },
                            Modifier.align(Alignment.TopEnd).padding(6.dp)) {
                            Icon(Icons.Filled.Close, "close showcase",
                                tint = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
                if (showMore) {
                    ModalBottomSheet(onDismissRequest = { showMore = false }) {
                        Column(Modifier.padding(bottom = 32.dp)) {
                            ListItem(headlineContent = { Text("Live Study") },
                                modifier = Modifier.clickable { tab = 10; showMore = false })
                            ListItem(headlineContent = { Text("Scope") },
                                modifier = Modifier.clickable { tab = 4; showMore = false })
                            ListItem(headlineContent = { Text("Detect") },
                                modifier = Modifier.clickable { tab = 5; showMore = false })
                            ListItem(headlineContent = { Text("Measure") },
                                modifier = Modifier.clickable { tab = 6; showMore = false })
                            ListItem(headlineContent = { Text("Trajectories") },
                                modifier = Modifier.clickable { tab = 7; showMore = false })
                            ListItem(headlineContent = { Text("Depth X-ray") },
                                modifier = Modifier.clickable { tab = 8; showMore = false })
                            ListItem(headlineContent = { Text("Glare Lab") },
                                modifier = Modifier.clickable { tab = 9; showMore = false })
                        }
                    }
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
            var shown by remember { mutableIntStateOf(0) }
            LaunchedEffect(statsOpen) {
                shown = 0
                if (statsOpen) repeat(4) { delay(160.milliseconds); shown++ }
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
    fun HomeScreen(onGo: (Int) -> Unit, onShowcase: () -> Unit = {}) {
        val map = remember { loadColonMap("colon_map2.bin") }
        var spin by remember { mutableFloatStateOf(0f) }
        LaunchedEffect(Unit) { while (true) { spin += 0.006f; delay(16.milliseconds) } }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Canvas(Modifier.fillMaxWidth().height(320.dp)
                .pointerInput(Unit) { detectTapGestures { onShowcase() } }) {
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
            Text("tap the model for the showcase",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline)

            val taglines = listOf(
                "the all-seeing monster" to Color(0xFF5FB5AD),

                "leaves no surface unmapped" to Color(0xFFE0796B),
                "one camera. one vision." to Color(0xFF3FD6FF),
                "98.6% agreement" to Color(0xFFDCA645))
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
                    "that system.", textAlign = TextAlign.Center)
            val mainCards = listOf(
                Triple(0, "Mission", "What this project set out to prove, and how it went."),
                Triple(1, "Results", "The numbers, measured against ground truth."),
                Triple(3, "Studies", "Reference runs in 3D, plus every study this phone has " +
                        "run on new footage — replayable frame by frame."),
                Triple(10, "Live Study", "The whole pipeline running as footage streams: " +
                        "the map, the coverage count, and any findings grow in real time."))

            val labCards = listOf(
                Triple(4, "Scope", "Point the camera at anything and watch a neural network " +
                        "judge its depth, live on this phone."),
                Triple(5, "Detect", "Real colonoscopy cases, analyzed on this phone. Find the polyp."),
                Triple(6, "Measure", "Tap two points on a colon frame and get the real distance " +
                        "in millimeters, from depth alone."),
                Triple(7, "Trajectories", "The camera path Argus computed, laid over the truth " +
                    "a robot recorded. Spin them together."),
                Triple(8, "Depth X-ray", "Generic AI versus the fine-tuned network, side by " +
                    "side. The whole project in one comparison."),
                Triple(9, "Glare Lab", "Watch a reflection detector pick apart real endoscopic glare"))

            mainCards.forEach { (idx, title, desc) ->
                Card(Modifier.fillMaxWidth().clickable { onGo(idx) }) {
                    Column(Modifier.padding(16.dp)) {
                        Text(title, style = MaterialTheme.typography.titleLarge,
                            color = MaterialTheme.colorScheme.primary)
                        Text(desc, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            Spacer(Modifier.height(8.dp))
            Text("LABORATORY", style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.outline)
            HorizontalDivider(Modifier.padding(horizontal = 40.dp), thickness = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant)
            Spacer(Modifier.height(8.dp))

            labCards.forEach { (idx, title, desc) ->
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
                                        SimpleDateFormat("MMM d HH.mm.ss", Locale.US).format(Date()))
                                d.mkdirs(); recDir = d; recCount = 0; recOn = true
                            }
                        }) { Text(if (recOn) "Stop  ●  $recCount" else "Record") }
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
                    var i by remember(d) { mutableIntStateOf(0) }
                    LaunchedEffect(d) { while (true) { delay(250.milliseconds); i = (i + 1) % n } }
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
            ExpandCard("What are Polyps?",
                "A polyp is a small growth on the wall of the colon. Most are harmless. Some " +
                        "turn into colorectal cancer over the years. Catch one early and remove it, " +
                        "and that cancer just never happens. That is the entire point of a screening " +
                        "colonoscopy.")
            ExpandCard("The Problem",
                "According to biomedical reports, around 1 in 4 precancerous polyps (adenomas)" +
                        "are missed during standard colonoscopies, which creates a significant risk for the patient." +
                        "The camera only shows a small window of a long, folded tunnel. Some patches " +
                        "of wall never make it into that window, and there is no way for the doctor " +
                        "to know which ones. This is a significant problem in modern colonoscopy.")
            ExpandCard("Our Goal",
                "The mission of this project is to figure out which parts of the colon were missed, using " +
                        "only the video from a normal endoscope. No extra sensors or new hardware.")
            ExpandCard("How ARGUS Works",
                "ARGUS consists of five steps.\n\n" +
                        "1. A neural network estimates how far away every pixel is.\n" +
                        "2. A tracking algorithm works out how the camera moved between frames.\n" +
                        "3. Thousands of those noisy snapshots get merged into one clean 3D surface.\n" +
                        "4. Every patch of that surface gets checked: if fewer than three frames saw " +
                        "it, it gets flagged as a blind spot.\n" +
                        "5. The polyp detector looks for polyps in every frame, then marks and " +
                        "measures what it finds.")
            ExpandCard("Setbacks",
                "The original idea was that lens glare was ruining the tracking. Testing " +
                        "proved that mostly wrong because the original SLAM system used, ORB-SLAM3," +
                        "already accounts for that issue, and adding a secondary glare-filtering layer " +
                        "actually worsened the outcome. Then the whole system broke when it switched to a " +
                        "generic AI depth model, and a controlled experiment traced the failure to " +
                        "that one part: the model had never seen the inside of a body.")
            ExpandCard("Discoveries",
                "After training the depth model on colon images, agreement with ground truth " +
                        "went from 74 percent to 98.6 percent. The takeaway is bigger than one app: " +
                        "AI vision cannot be trusted on territory it was never taught, and one " +
                        "targeted fix in the right place can rescue a whole system.")
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
                    Text("Depth, before and after fine-tuning",
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
                "0.60 mm" to "median measurement error on polyp sized targets",
                "11 of 21" to "unseen polyp videos flagged at thresholds fixed in advance — " +
                        "while 17 of 20 control videos stayed clean, and every flagged " +
                        "control turned out to be real non-polyp pathology",
                "9×" to "the tracking-error cost of an actively deforming colon, measured " +
                        "against a rigid twin of the same phantom with ground-truth " +
                        "camera paths").forEach { (n, d) ->
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
                   val trajN: Int, val traj: FloatArray, val lesions: List<FloatArray>,
                   val views: IntArray? = null, val spanMm: Float = 0f,
                   val mesh: MeshData? = null, val centerMm: FloatArray? = null)

    private fun loadColonMap(file: String): ColonMap {
        val bb = java.nio.ByteBuffer.wrap(assets.open(file).readBytes())
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
        val lesions = mutableListOf<FloatArray>()
        if (bb.remaining() >= 4) {
            val numLesions = bb.int
            repeat(numLesions) { lesions.add(floatArrayOf(bb.float, bb.float, bb.float, bb.float)) }
        }
        return ColonMap(n, xyz, r, g, b, blind, born, m, traj, lesions)
    }

    class Anchor(val x: Float, val y: Float, val z: Float, val title: String, val body: String)

    private fun studyReport(fr: StudyEngine.StudyResult): String {
        val when0 = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())
        val sb = StringBuilder()
        sb.appendLine("ARGUS live study report — $when0")
        sb.appendLine("source: ${StudyRunner.sourceName}")
        sb.appendLine("frames analyzed: ${fr.frames}")
        sb.appendLine("surface patches mapped: ${fr.voxels}")
        sb.appendLine("wall seen fewer than 3 times in the 80 analyzed frames: %.1f%%"
            .format(fr.blindPct))
        sb.appendLine("  (a harsher statistic than the validated coverage numbers: short " +
                "clip, sampled frames, uncalibrated camera — describes this run, not the colon)")
        sb.appendLine("camera path: %.0f mm (estimated, uncalibrated)".format(fr.pathMm))
        sb.appendLine("detector firings: ${fr.detectionFrames} of ${fr.frames} frames")
        if (fr.lesions.isEmpty())
            sb.appendLine("findings: none passed the persistence filter")
        else fr.lesions.forEachIndexed { i, l ->
            sb.appendLine("finding ${i + 1}: polyp candidate ~%.1f mm, %d supporting frames, avg confidence %.2f"
                .format(l.sizeMm, l.support, l.avgConf))
        }
        sb.appendLine("stage timing: depth %.1fs (%.1f fps) | tracking %.1fs (%.1f fps) | fusion %.1fs | detection %.1fs (%.1f fps)"
            .format(fr.stageSecs[0], fr.frames / maxOf(0.01f, fr.stageSecs[0]),
                fr.stageSecs[1], fr.frames / maxOf(0.01f, fr.stageSecs[1]), fr.stageSecs[2],
                fr.stageSecs[3], fr.frames / maxOf(0.01f, fr.stageSecs[3])))
        sb.appendLine("total: %.0f s on-device (%.2f frames/s end-to-end)"
            .format(fr.secondsTaken, fr.frames / fr.secondsTaken))
        sb.appendLine("note: computed on this device from the chosen video; distances are " +
                "uncalibrated estimates; detection settings fixed in advance.")
        return sb.toString()
    }

    // A finished on-device study, reshaped into the same structure the reference runs use,
    // so the whole Studies screen (canvas, replay, anchors) works on it unchanged.
    private fun colonMapFromStudy(sr: StudyEngine.StudyResult): ColonMap {
        val total = sr.pts.size / 3
        val stride = maxOf(1, total / 6000)
        val idx = (0 until total step stride).toList()
        val n = idx.size
        val xyz = FloatArray(3 * n)
        val r = IntArray(n); val g = IntArray(n); val b = IntArray(n)
        val blind = BooleanArray(n); val born = IntArray(n)
        val vcs = IntArray(n)
        for ((j, i) in idx.withIndex()) {
            xyz[3 * j] = sr.pts[3 * i]; xyz[3 * j + 1] = sr.pts[3 * i + 1]
            xyz[3 * j + 2] = sr.pts[3 * i + 2]
            r[j] = (sr.colors[i] shr 16) and 0xFF
            g[j] = (sr.colors[i] shr 8) and 0xFF
            b[j] = sr.colors[i] and 0xFF
            blind[j] = sr.blind[i]; born[j] = sr.born[i]
            vcs[j] = sr.viewCounts?.getOrNull(i) ?: if (sr.blind[i]) 1 else 4
        }
        val lesions = sr.lesions.map { floatArrayOf(it.x, it.y, it.z, it.sizeMm) }
        return ColonMap(n, xyz, r, g, b, blind, born, sr.frames, sr.traj, lesions,
            vcs, sr.spanMm, sr.mesh, sr.centerMm)
    }

    private fun makeLiveAnchors(map: ColonMap, sr: StudyEngine.StudyResult?): List<Anchor> {
        if (map.n == 0 || sr == null) return emptyList()
        val out = mutableListOf<Anchor>()
        val bi = map.blind.indexOfFirst { it }
        if (bi >= 0) out.add(Anchor(map.xyz[3 * bi], map.xyz[3 * bi + 1], map.xyz[3 * bi + 2],
            "A blind spot",
            "This patch was seen fewer than three times in the analyzed footage. " +
                    "Every red point on this model is a place the camera never properly inspected."))
        sr.lesions.forEach { l ->
            out.add(Anchor(l.x, l.y, l.z, "Polyp candidate — found in this study",
                "Flagged in ${l.support} separate frames, each within ten millimeters of " +
                        "this spot, average confidence %.2f. Estimated size about %.1f mm — "
                            .format(l.avgConf, l.sizeMm) +
                        "an estimate, because this camera is uncalibrated."))
        }
        return out
    }

    private fun makeAnchors(map: ColonMap): List<Anchor> {
        if (map.n == 0) return emptyList()
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
        val base = mutableListOf(
            Anchor(map.xyz[3 * bi], map.xyz[3 * bi + 1], map.xyz[3 * bi + 2], "A blind spot",
                "The camera never got a proper look at this patch. Fewer than three frames saw " +
                        "it. In a real procedure, this is exactly where a polyp could hide unnoticed."),
            Anchor(map.xyz[3 * si], map.xyz[3 * si + 1], map.xyz[3 * si + 2], "Where the run began",
                "The scope entered here. Every point in this model was placed by combining two " +
                        "guesses: how far away each pixel was, and how the camera was moving."))
        map.lesions.forEach { lesion ->
            base.add(Anchor(lesion[0], lesion[1], lesion[2], "Polyp — found by the system",
                "The detector flagged this spot in 21 separate frames, and every flag landed " +
                        "within ten millimeters of the same 3D location. Estimated size about " +
                        "${lesion[3]} mm. On real footage, with no ground truth given, the full pipeline " +
                        "found it, placed it, and measured it."))
        }
        return base
    }

    @Composable
    fun MapScreen() {
        var run by remember { mutableIntStateOf(0) }
        val live = StudyRunner.result
        val refMesh = remember { runCatching {
            loadMeshBin(assets.open("mesh_cecum.bin").readBytes()) }.getOrNull() }
        val refMeshHk = remember { runCatching {
            loadMeshBin(assets.open("mesh_hk.bin").readBytes()) }.getOrNull() }
        val refMeshFull = remember { runCatching {
            loadMeshBin(assets.open("mesh_full.bin").readBytes()) }.getOrNull() }
        var surfaceMode by remember(run) { mutableStateOf(true) }
        val map = remember(run, live) {
            when (run) {
                0 -> loadColonMap("colon_map2.bin").let { base ->
                    if (refMesh != null) {
                        val cs = meshCenterSpan(refMesh)
                        ColonMap(base.n, base.xyz, base.r, base.g, base.b, base.blind,
                            base.born, base.trajN, base.traj, base.lesions,
                            base.views, cs[3], refMesh,
                            floatArrayOf(cs[0], cs[1], cs[2]))
                    } else base
                }
                1 -> loadColonMap("hk_map.bin").let { base ->
                    if (refMeshHk != null) {
                        val cs = meshCenterSpan(refMeshHk)
                        ColonMap(base.n, base.xyz, base.r, base.g, base.b, base.blind,
                            base.born, base.trajN, base.traj, base.lesions,
                            base.views, cs[3], refMeshHk,
                            floatArrayOf(cs[0], cs[1], cs[2]))
                    } else base
                }
                3 -> if (refMeshFull != null) {
                    val cs = meshCenterSpan(refMeshFull)
                    ColonMap(0, FloatArray(0), IntArray(0), IntArray(0), IntArray(0),
                        BooleanArray(0), IntArray(0), 0, FloatArray(0), emptyList(),
                        null, cs[3], refMeshFull, floatArrayOf(cs[0], cs[1], cs[2]))
                } else ColonMap(0, FloatArray(0), IntArray(0), IntArray(0), IntArray(0),
                    BooleanArray(0), IntArray(0), 0, FloatArray(0), emptyList())
                else -> live?.let { colonMapFromStudy(it) }
                    ?: ColonMap(0, FloatArray(0), IntArray(0), IntArray(0), IntArray(0),
                        BooleanArray(0), IntArray(0), 0, FloatArray(0), emptyList())
            }
        }
        val anchors = remember(run, live) {
            if (run == 2) makeLiveAnchors(map, live) else makeAnchors(map)
        }
        var yaw by remember { mutableFloatStateOf(0.6f) }
        var pitch by remember { mutableFloatStateOf(0.3f) }
        var zoom by remember { mutableFloatStateOf(1f) }
        var showBlind by remember { mutableStateOf(false) }
        var heatMode by remember(run) { mutableStateOf(run >= 2) }
        var measureMode by remember(run) { mutableStateOf(false) }
        var mSel by remember(run) { mutableStateOf(listOf<Int>()) }
        var replay by remember { mutableStateOf(false) }
        var playing by remember { mutableStateOf(false) }
        var t by remember { mutableFloatStateOf(0f) }
        var picked by remember { mutableStateOf<Anchor?>(null) }
        var open by remember { mutableIntStateOf(1) }   // which section is expanded
        var canvasW by remember { mutableFloatStateOf(1f) }
        var canvasH by remember { mutableFloatStateOf(1f) }
        val pulse by rememberInfiniteTransition(label = "p")
            .animateFloat(10f, 26f, infiniteRepeatable(
                animation = tween(900),
                repeatMode = RepeatMode.Reverse), label = "p")

        fun project(x: Float, y: Float, z: Float, cy: Float, sy: Float,
                    cp: Float, sp: Float): Triple<Float, Float, Float> {
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

        Column(Modifier.fillMaxSize()) {

            // ---------- THE MODEL: fixed, always visible, never covered ----------
            Box(Modifier.fillMaxWidth().height(330.dp)) {
                Canvas(Modifier.fillMaxSize()
                    .pointerInput(Unit) {
                        detectTransformGestures { _, pan, gz, _ ->
                            yaw += pan.x * 0.008f; pitch += pan.y * 0.008f
                            zoom = (zoom * gz).coerceIn(0.5f, 4f)
                        }
                    }
                    .pointerInput(run) {
                        detectTapGestures { tap ->
                            val cy = cos(yaw); val sy = sin(yaw)
                            val cp = cos(pitch); val sp = sin(pitch)
                            if (measureMode) {
                                // pick the nearest model point: Measure, on the map itself
                                var best = -1; var bestD = 45f * 45f
                                for (i in 0 until map.n) {
                                    val (px, py, _) = project(map.xyz[3 * i],
                                        map.xyz[3 * i + 1], map.xyz[3 * i + 2],
                                        cy, sy, cp, sp)
                                    val dx = tap.x - px; val dy = tap.y - py
                                    val d = dx * dx + dy * dy
                                    if (d < bestD) { bestD = d; best = i }
                                }
                                if (best >= 0)
                                    mSel = (mSel + best).takeLast(2)
                            } else picked = anchors.firstOrNull {
                                val (px, py, _) = project(it.x, it.y, it.z, cy, sy, cp, sp)
                                val dx = tap.x - px; val dy = tap.y - py
                                dx * dx + dy * dy < 55f * 55f
                            }
                        }
                    }) {
                    canvasW = size.width; canvasH = size.height
                    val cy = cos(yaw); val sy = sin(yaw)
                    val cp = cos(pitch); val sp = sin(pitch)
                    val blindPct = map.blind.count { it }.toFloat() / maxOf(1, map.n)
                    val rare = blindPct < 0.25f
                    // Fused surface when available (and not replaying the build animation);
                    // otherwise points in painter's order.
                    val meshOk = surfaceMode && map.mesh != null && map.centerMm != null &&
                            map.spanMm > 0f && !replay
                    if (meshOk) {
                        val cmm = map.centerMm!!
                        drawMeshLit(map.mesh!!, cmm[0], cmm[1], cmm[2], map.spanMm,
                            yaw, pitch, zoom,
                            if (heatMode) 1 else if (showBlind) 2 else 0, 3)
                    }
                    // Painter's order: far wall drawn first so the near wall covers it —
                    // this is what makes the cloud read as a tube instead of a blob.
                    val pxA = FloatArray(map.n); val pyA = FloatArray(map.n)
                    val ppA = FloatArray(map.n) { -1f }
                    for (i in 0 until map.n) {
                        if (meshOk) break
                        if (replay && map.born[i] > t) continue
                        val (px, py, persp) = project(map.xyz[3 * i], map.xyz[3 * i + 1],
                            map.xyz[3 * i + 2], cy, sy, cp, sp)
                        pxA[i] = px; pyA[i] = py; ppA[i] = persp
                    }
                    val order = (0 until map.n).sortedBy { ppA[it] }
                    for (i in order) {
                        val persp = ppA[i]
                        if (persp <= 0f) continue
                        val shade = (0.45f + 0.75f * persp / zoom).coerceIn(0.35f, 1f)
                        // When blind spots are rare, emphasize them (bigger, dim the rest).
                        // When they dominate, draw both classes equally so the picture
                        // matches the percentage instead of exaggerating it.
                        // splat-sized points fuse into a surface instead of reading as dots
                        if (heatMode) {
                            val vc = map.views?.getOrNull(i) ?: if (map.blind[i]) 1 else 4
                            drawCircle(coverageColor(minOf(vc, 8) / 8f, shade),
                                4.2f * persp.coerceIn(0.6f, 1.6f),
                                Offset(pxA[i], pyA[i]), alpha = 0.95f)
                        } else if (showBlind && map.blind[i]) {
                            drawCircle(Color(0.90f * shade, 0.23f * shade, 0.27f * shade),
                                if (rare) 5.5f else 4.2f * persp.coerceIn(0.6f, 1.6f),
                                Offset(pxA[i], pyA[i]))
                        } else {
                            drawCircle(Color(map.r[i] / 255f * shade, map.g[i] / 255f * shade,
                                map.b[i] / 255f * shade),
                                4.2f * persp.coerceIn(0.6f, 1.6f), Offset(pxA[i], pyA[i]),
                                alpha = if (showBlind && rare) 0.30f else 0.95f)
                        }
                    }
                    val upTo = (if (replay) t.toInt() else map.trajN - 1)
                        .coerceAtMost(map.trajN - 1)
                    for (j in 1..upTo) {
                        val (ax, ay, _) = project(map.traj[3 * (j - 1)], map.traj[3 * (j - 1) + 1],
                            map.traj[3 * (j - 1) + 2], cy, sy, cp, sp)
                        val (bx, by, _) = project(map.traj[3 * j], map.traj[3 * j + 1],
                            map.traj[3 * j + 2], cy, sy, cp, sp)
                        drawLine(Color(0xFF3FD6FF), Offset(ax, ay), Offset(bx, by), 3f)
                    }
                    if (measureMode && mSel.isNotEmpty()) {
                        val ends = mSel.map { i ->
                            val (px, py, _) = project(map.xyz[3 * i], map.xyz[3 * i + 1],
                                map.xyz[3 * i + 2], cy, sy, cp, sp)
                            Offset(px, py)
                        }
                        ends.forEach { drawCircle(Color(0xFF3FD6A0), 8f, it) }
                        if (ends.size == 2)
                            drawLine(Color(0xFF3FD6A0), ends[0], ends[1], 4f)
                    }
                    anchors.forEach {
                        val gold = it.title.startsWith("Polyp")
                        if (replay && (!gold || t < 10f)) return@forEach
                        val (px, py, _) = project(it.x, it.y, it.z, cy, sy, cp, sp)
                        if (gold && replay) drawCircle(Color(0xFFDCA645), pulse,
                            Offset(px, py), alpha = 0.35f)
                        drawCircle(if (gold) Color(0xFFDCA645) else Color(0xFF5FB5AD),
                            if (gold) 13f else 10f, Offset(px, py), alpha = 0.95f)
                        drawCircle(Color(0xFF121519), if (gold) 6f else 5f, Offset(px, py))
                    }
                }
                Text("drag · pinch · tap markers", Modifier.align(Alignment.TopEnd).padding(8.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (map.trajN > 1) TextButton(onClick = {
                    // align the view along the camera's overall direction of travel —
                    // the tunnel view, where a colon looks like a colon
                    val li = 3 * (map.trajN - 1)
                    val dx = map.traj[li] - map.traj[0]
                    val dy = map.traj[li + 1] - map.traj[1]
                    val dz = map.traj[li + 2] - map.traj[2]
                    val ny = kotlin.math.atan2(-dx, dz)
                    val z1 = -dx * sin(ny) + dz * cos(ny)
                    yaw = ny; pitch = kotlin.math.atan2(dy, z1); zoom = 1.3f
                }, Modifier.align(Alignment.TopStart)) { Text("down the tube") }
                Row(Modifier.align(Alignment.BottomStart)) {
                    if (map.mesh != null) TextButton(onClick = { surfaceMode = !surfaceMode }) {
                        Text(if (surfaceMode) "point cloud" else "surface")
                    }
                    TextButton(onClick = { heatMode = !heatMode }) {
                        Text(if (heatMode) "tissue colors" else "coverage heat")
                    }
                    TextButton(onClick = {
                        measureMode = !measureMode; mSel = listOf(); picked = null
                    }) { Text(if (measureMode) "done measuring" else "measure") }
                }
                if (measureMode) {
                    val label = if (mSel.size == 2) {
                        val a = mSel[0]; val b = mSel[1]
                        val dx = map.xyz[3*a] - map.xyz[3*b]
                        val dy = map.xyz[3*a+1] - map.xyz[3*b+1]
                        val dz = map.xyz[3*a+2] - map.xyz[3*b+2]
                        val d = sqrt(dx*dx + dy*dy + dz*dz)
                        if (map.spanMm > 0f) "≈ %.1f mm (estimated)".format(d * map.spanMm)
                        else "%.2f model units (no scale on this replay)".format(d)
                    } else "tap two points on the model"
                    Card(Modifier.align(Alignment.BottomCenter).padding(bottom = 44.dp)) {
                        Text(label, Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            style = MaterialTheme.typography.labelLarge)
                    }
                }
                if (run == 2 && map.n == 0)
                    Text(if (StudyRunner.running) "Study in progress — the map appears here " +
                            "when it finishes." else "No live study yet. Open \"Run the " +
                            "system\" below and feed the pipeline a video.",
                        Modifier.align(Alignment.Center).padding(24.dp),
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                picked?.let { a ->
                    Card(Modifier.align(Alignment.BottomCenter).padding(8.dp)) {
                        Column(Modifier.padding(12.dp)) {
                            Text(a.title, style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.primary)
                            Text(a.body, style = MaterialTheme.typography.bodySmall)
                            TextButton(onClick = { picked = null }) { Text("close") }
                        }
                    }
                }
            }

            // ---------- REPLAY STRIP: appears only during a run ----------
            if (replay && run < 2) {
                val fMax = if (run == 0) 91 else 59
                val dir = if (run == 0) "replay" else "hkreplay"
                val frameIdx = (t / map.trajN * fMax).toInt().coerceIn(0, fMax)
                val raw = remember(run, frameIdx) {
                    BitmapFactory.decodeStream(assets.open("$dir/f%03d.jpg".format(frameIdx))) }
                val dep = remember(run, frameIdx) {
                    BitmapFactory.decodeStream(assets.open("$dir/f%03dd.jpg".format(frameIdx))) }
                Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Image(raw.asImageBitmap(), null, Modifier.weight(1f).height(86.dp))
                    Image(dep.asImageBitmap(), null, Modifier.weight(1f).height(86.dp))
                    Column(Modifier.weight(1.2f)) {
                        Text("frame ${t.toInt()}/${map.trajN}",
                            style = MaterialTheme.typography.labelLarge)
                        Text("${map.born.count { it <= t }} points placed",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (run == 1 && t >= 10f)
                            Text("POLYP DETECTED", style = MaterialTheme.typography.labelLarge,
                                color = Color(0xFFDCA645))
                    }
                }
            }
            if (replay && run == 2 && live != null) {
                val frameIdx = t.toInt().coerceIn(0, live.frames - 1)
                Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Image(live.thumbs[frameIdx].asImageBitmap(), null,
                        Modifier.weight(1f).height(86.dp))
                    Column(Modifier.weight(1.4f)) {
                        Text("frame ${frameIdx + 1}/${live.frames}",
                            style = MaterialTheme.typography.labelLarge)
                        Text("${live.frameVoxels[frameIdx]} patches · " +
                                "%.1f%% under-seen".format(live.frameBlindPct[frameIdx]),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(StudyRunner.sourceName, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (live.lesions.any { it.frame <= frameIdx })
                            Text("POLYP PINNED", style = MaterialTheme.typography.labelLarge,
                                color = Color(0xFFDCA645))
                    }
                }
            }
            if (replay && map.trajN > 1) {
                Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    FilledTonalButton(onClick = {
                        if (!playing && t >= map.trajN - 1) t = 0f
                        playing = !playing
                    }) { Text(if (playing) "Pause" else "Play") }
                    Slider(value = t, onValueChange = { playing = false; t = it },
                        valueRange = 0f..(map.trajN - 1).toFloat(),
                        modifier = Modifier.weight(1f))
                }
            }

            // ---------- THE STORY: scrolls under the model, one section open at a time ----------
            Column(Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState())
                .padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {

                @Composable
                fun Section(idx: Int, title: String, content: @Composable () -> Unit) {
                    Card(Modifier.fillMaxWidth().clickable { open = if (open == idx) -1 else idx }) {
                        Column(Modifier.padding(14.dp)) {
                            Text(title, style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.primary)
                            androidx.compose.animation.AnimatedVisibility(open == idx) {
                                Column(Modifier.padding(top = 8.dp)) { content() }
                            }
                        }
                    }
                }

                Section(0, "The goal") {
                    Text("Around one in four precancerous polyps is missed in a standard " +
                            "colonoscopy, usually because part of the wall was never actually " +
                            "looked at. This screen shows the complete answer: a 3D map built " +
                            "from the video alone, the spots that were never inspected, and any " +
                            "polyps that were found along the way.")
                }
                Section(1, "Run the system") {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilterChip(selected = run == 0, onClick = {
                            run = 0; replay = false; playing = false; t = 0f },
                            label = { Text("Phantom (validated)") })
                        FilterChip(selected = run == 1, onClick = {
                            run = 1; replay = false; playing = false; t = 0f },
                            label = { Text("Real footage") })
                    }
                    FilterChip(selected = run == 3, onClick = {
                        run = 3; replay = false; playing = false; t = 0f },
                        label = { Text("Full withdrawal (screening speed)") })
                    FilterChip(selected = run == 2, onClick = {
                        run = 2; replay = false; playing = false; t = 0f },
                        label = { Text("New footage — run a live study") })
                    if (run == 2) {
                        StudySourcePanel()
                        Spacer(Modifier.height(6.dp))
                    }
                    Button(onClick = {
                        replay = !replay
                        if (replay) { t = 0f; playing = true; open = 1 } else playing = false
                    }, enabled = map.trajN > 1) {
                        Text(if (replay) "Back to explore" else "Watch it build") }
                    if (replay && !playing && map.trajN > 1 && t >= map.trajN - 1) {
                        Text(when (run) {
                            0 -> "276 frames fused into one surface. 94.7 percent of the wall was " +
                                    "properly inspected. No polyps here: this phantom is clean, and " +
                                    "saying so is the honest half of detection."
                            1 -> "Real colonoscopy, 60 frames. The system tracked the camera, " +
                                    "built this surface, and found one polyp of about 6 mm. Its gold " +
                                    "marker is on the model above."
                            else -> live?.let {
                                "${it.frames} frames of ${StudyRunner.sourceName} fused into " +
                                        "${it.voxels} surface patches. %.1f%% of the mapped wall "
                                            .format(it.blindPct) +
                                        "was seen fewer than three times. " +
                                        if (it.lesions.isEmpty()) "No polyp met the detection bar."
                                        else "One or more polyp candidates were pinned — gold " +
                                                "markers above."
                            } ?: ""
                        }, Modifier.padding(top = 8.dp))
                    }
                }
                Section(2, "Blind spots") {
                    val pct = if (run == 3 && map.mesh != null)
                        100f * map.mesh!!.views.count { it < 3 } /
                                maxOf(1, map.mesh!!.views.size)
                    else 100f * map.blind.count { it } / maxOf(1, map.n)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Switch(checked = showBlind, onCheckedChange = {
                            showBlind = it; if (it) open = 2 })
                        Text("  show on the model")
                    }
                    Text("%.1f%% of this surface was seen fewer than three times — glimpsed, "
                        .format(pct) +
                            "never inspected. In a real procedure these are the patches a doctor " +
                            "would be sent back to check.")
                }
                Section(3, "Findings") {
                    if (run == 3) {
                        Text("This reference shows what coverage looks like at genuine " +
                                "screening speed: a complete withdrawal performed by a " +
                                "gastroenterologist, mapped by the pipeline. Nearly half " +
                                "the mapped wall got fewer than three proper looks — " +
                                "the problem this project exists to make visible.")
                    } else if (run == 2) {
                        val fr = live
                        if (fr == null)
                            Text("No live study yet. Open \"Run the system\" and feed the " +
                                    "pipeline a video.")
                        else {
                            Text("Source: ${StudyRunner.sourceName}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                            if (fr.lesions.isEmpty())
                                Text("No polyp met the detection bar (confidence at least 0.7, " +
                                        "plausible size, repeated hits at one 3D location). The " +
                                        "detector fired on ${fr.detectionFrames} of ${fr.frames} " +
                                        "frames without that persistence.")
                            else fr.lesions.forEach { l ->
                                Text("Polyp candidate — about %.1f mm".format(l.sizeMm),
                                    style = MaterialTheme.typography.labelLarge,
                                    color = Color(0xFFDCA645))
                                Text("Flagged in ${l.support} frames at the same 3D location, " +
                                        "average confidence %.2f. Size is an estimate — this "
                                            .format(l.avgConf) +
                                        "camera is uncalibrated.")
                            }
                        }
                    } else if (map.lesions.isEmpty())
                        Text("No polyps detected in this run. On a clean phantom, that is " +
                                "the correct answer.")
                    else map.lesions.forEach { lesion ->
                        Text("Polyp — about ${lesion[3]} mm", style = MaterialTheme.typography.labelLarge,
                            color = Color(0xFFDCA645))
                        Text("Flagged in 21 separate frames, every flag within ten " +
                                "millimeters of the same 3D spot. Tap its gold marker above for " +
                                "the full story.")
                    }
                }
                Section(4, "The numbers") {
                    if (run == 3) {
                        Text("Complete withdrawal, camera driven by a gastroenterologist, " +
                                "camera path recorded as ground truth.\n" +
                                "Frames analyzed: 700 (every 12th of 8,400)\n" +
                                "Tracking error: 150 mm over a 4,299 mm path (3.5%)\n" +
                                "Wall seen fewer than 3 times at real speed: 44.4%\n" +
                                "The same pipeline, the same settings — only the footage " +
                                "is longer and the motion is human.")
                    } else if (run == 2) {
                        val fr = live
                        if (fr == null)
                            Text("Numbers appear here once a live study has run.")
                        else {
                            Text("Frames analyzed: ${fr.frames}\n" +
                                    "Surface patches mapped: ${fr.voxels}\n" +
                                    "Wall seen fewer than 3 times in the 80 analyzed frames: %.1f%%\n".format(fr.blindPct) +
                                    "Camera path: %.0f mm (estimated)\n".format(fr.pathMm) +
                                    "Computed in %.0f s on this device (%.2f frames/s end-to-end)"
                                        .format(fr.secondsTaken, fr.frames / fr.secondsTaken))
                            Text("Stage timing — depth %.1f s (%.1f fps) · tracking %.1f s (%.1f fps) · fusion %.1f s · detection %.1f s (%.1f fps)"
                                .format(fr.stageSecs[0], fr.frames / maxOf(0.01f, fr.stageSecs[0]),
                                    fr.stageSecs[1], fr.frames / maxOf(0.01f, fr.stageSecs[1]),
                                    fr.stageSecs[2],
                                    fr.stageSecs[3], fr.frames / maxOf(0.01f, fr.stageSecs[3])),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Text("Every number here was computed from the chosen video during " +
                                    "this run. Distances are estimates: the camera is " +
                                    "uncalibrated, and detection settings are fixed in advance, " +
                                    "never adjusted to the video.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                            var showReport by remember(fr) { mutableStateOf(false) }
                            val clip = LocalClipboardManager.current
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                FilledTonalButton(onClick = { showReport = !showReport }) {
                                    Text(if (showReport) "Hide report" else "Show report")
                                }
                                if (showReport) FilledTonalButton(onClick = {
                                    clip.setText(AnnotatedString(studyReport(fr)))
                                }) { Text("Copy") }
                            }
                            if (showReport) OutlinedTextField(
                                value = studyReport(fr), onValueChange = {}, readOnly = true,
                                textStyle = MaterialTheme.typography.bodySmall.copy(
                                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace),
                                modifier = Modifier.fillMaxWidth().height(230.dp))
                        }
                    } else
                        Text("Blind-spot agreement with ground truth: 98.6 percent\n" +
                                "Camera tracking error: 0.91 mm average\n" +
                                "Surface accuracy: 1.3 to 2.4 mm\n" +
                                "All measured against a phantom whose true shape and camera path " +
                                "were recorded by machines.")
                }
                Section(5, "How it works") {
                    Text("A neural network reads depth from each flat frame. A tracking " +
                            "algorithm recovers the camera's motion. Thousands of noisy " +
                            "snapshots vote themselves into one clean surface. Every patch " +
                            "then reports how many times it was actually seen — and the polyp " +
                            "detector marks anything it finds along the way.")
                }
            }
        }
    }

    class MeasureSet(val img: Bitmap, val w: Int, val h: Int, val fx: Float, val fy: Float,
                     val cx: Float, val cy: Float, val depth: ShortArray)

        private fun loadMeasure(j: Int): MeasureSet {
            val img = BitmapFactory.decodeStream(assets.open("measure/m$j.jpg"))
            val bb = java.nio.ByteBuffer.wrap(assets.open("measure/m$j.bin").readBytes())
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            val w = bb.int; val h = bb.int
            val fx = bb.float; val fy = bb.float; val cx = bb.float; val cy = bb.float
            val d = ShortArray(w * h); bb.asShortBuffer().get(d)
            return MeasureSet(img, w, h, fx, fy, cx, cy, d)
        }

        private fun depthAt(s: MeasureSet, u: Int, v: Int): Float {
            val vals = ArrayList<Float>()
            for (dy in -2..2) for (dx in -2..2) {
                val x = u + dx; val y = v + dy
                if (x in 0 until s.w && y in 0 until s.h) {
                    val raw = s.depth[y * s.w + x].toInt() and 0xFFFF
                    if (raw > 0) vals.add(raw * 100f / 65535f / 1000f)
                }
            }
            if (vals.isEmpty()) return 0f
            vals.sort(); return vals[vals.size / 2]
        }

        @Composable
        fun MeasureScreen() {
            var preset by remember { mutableIntStateOf(0) }
            val set = remember(preset) { loadMeasure(preset) }
            var pts by remember(preset) { mutableStateOf(listOf<Pair<Float, Float>>()) }
            var mm by remember(preset) { mutableStateOf<Float?>(null) }
            Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Measure", style = MaterialTheme.typography.headlineMedium)
                Text("Tap two points. The distance between them comes from the depth network, " +
                        "the same math validated to under 1.3 millimeters against ground truth.")
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        repeat(2) { j -> FilterChip(selected = preset == j, onClick = { preset = j },
                            label = { Text("frame ${j + 1}") }) }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        repeat(2) { j -> FilterChip(selected = preset == j + 2, onClick = { preset = j + 2 },
                            label = { Text("frame ${j + 3}") }) }
                    }
                }
                Box(Modifier.fillMaxWidth().aspectRatio(set.w.toFloat() / set.h)) {
                    Image(set.img.asImageBitmap(), null, Modifier.fillMaxSize())
                    Canvas(Modifier.fillMaxSize().pointerInput(preset) {
                        detectTapGestures { tap ->
                            val sx = size.width.toFloat() / set.w
                            val p = Pair(tap.x / sx, tap.y / sx)
                            pts = if (pts.size >= 2) listOf(p) else pts + p
                            mm = null
                            if (pts.size == 2) {
                                val (a, b) = pts
                                val z1 = depthAt(set, a.first.toInt(), a.second.toInt())
                                val z2 = depthAt(set, b.first.toInt(), b.second.toInt())
                                if (z1 > 0 && z2 > 0) {
                                    val x1 = (a.first - set.cx) / set.fx * z1
                                    val y1 = (a.second - set.cy) / set.fy * z1
                                    val x2 = (b.first - set.cx) / set.fx * z2
                                    val y2 = (b.second - set.cy) / set.fy * z2
                                    mm = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)) * 1000f
                                }
                            }
                        }
                    }) {
                        val sx = size.width / set.w
                        pts.forEach { drawCircle(Color(0xFF3FD6A0), 9f,
                            Offset(it.first * sx, it.second * sx)) }
                        if (pts.size == 2) drawLine(Color(0xFF3FD6A0),
                            Offset(pts[0].first * sx, pts[0].second * sx),
                            Offset(pts[1].first * sx, pts[1].second * sx), 4f)
                    }
                }
                Text(mm?.let { "%.1f mm".format(it) } ?: if (pts.size == 1) "one more point…"
                else "tap the image",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.primary)
            }
        }

        @Composable
        fun TrajScreen() {
            val seqs = listOf("cecum_t1_a", "desc_t4_a", "trans_t1_a", "trans_t2_a", "trans_t2_b")
            var si by remember { mutableIntStateOf(0) }
            val data = remember(si) {
                val bb = java.nio.ByteBuffer.wrap(assets.open("traj/${seqs[si]}.bin").readBytes())
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val n = bb.int; val gTraj = FloatArray(3 * n); for (i in 0 until 3 * n) gTraj[i] = bb.float
            val m = bb.int; val fTraj = FloatArray(3 * m); for (i in 0 until 3 * m) fTraj[i] = bb.float
            Pair(gTraj, fTraj)
            }
            var yaw by remember { mutableFloatStateOf(0.5f) }
            var pitch by remember { mutableFloatStateOf(0.3f) }
            var zoom by remember { mutableFloatStateOf(1f) }
            Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Trajectories", style = MaterialTheme.typography.headlineMedium)
                Text("Amber is the true camera path, recorded by a robot. Teal is what Argus " +
                        "computed from the video alone. On average they differ by 0.91 millimeters.")
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        seqs.take(3).forEachIndexed { j, s -> FilterChip(selected = si == j,
                            onClick = { si = j }, label = { Text(s.take(6)) }) }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        seqs.drop(3).forEachIndexed { j, s -> FilterChip(selected = si == j + 3,
                            onClick = { si = j + 3 }, label = { Text(s.take(6)) }) }
                    }
                }
                Canvas(Modifier.fillMaxWidth().weight(1f).pointerInput(Unit) {
                    detectTransformGestures { _, pan, gz, _ ->
                        yaw += pan.x * 0.008f; pitch += pan.y * 0.008f
                        zoom = (zoom * gz).coerceIn(0.5f, 4f)
                    }
                }) {
                    val cy = cos(yaw); val sy = sin(yaw); val cp = cos(pitch); val sp = sin(pitch)
                    val s = size.minDimension * 0.42f * zoom
                    fun prj(a: FloatArray, i: Int): Offset {
                        val x = a[3*i]; val y = a[3*i+1]; val z = a[3*i+2]
                        val x1 = x*cy + z*sy; val z1 = -x*sy + z*cy
                        val y1 = y*cp - z1*sp
                        return Offset(size.width/2 + x1*s, size.height/2 + y1*s)
                    }
                    val (gTraj, fTraj) = data
                for (i in 1 until gTraj.size/3) drawLine(Color(0xFFDCA645), prj(gTraj, i-1), prj(gTraj, i), 5f)
                for (i in 1 until fTraj.size/3) drawLine(Color(0xFF5FB5AD), prj(fTraj, i-1), prj(fTraj, i), 3f)
                }
                Text("drag to turn · pinch to zoom", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        @Composable
        fun XRayScreen() {
            var preset by remember { mutableIntStateOf(0) }
            val strip = remember(preset) {
                BitmapFactory.decodeStream(assets.open("xray/x$preset.jpg")) }
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Depth X-ray", style = MaterialTheme.typography.headlineMedium)
                Text("The same frame, judged twice. The middle panel is a generic depth AI that " +
                        "learned from streets and living rooms. The right panel is the same kind of " +
                        "network after it was taught what a colon looks like. This comparison is what " +
                        "makes the whole system possible.")
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    repeat(4) { j -> FilterChip(selected = preset == j, onClick = { preset = j },
                        label = { Text("case ${j + 1}") }) }
                }
                Image(strip.asImageBitmap(), null, Modifier.fillMaxWidth())
                Text("raw  ·  generic AI  ·  fine-tuned",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.primary)
                Text("Comparisons precomputed from both models.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

    // Highlight pixels that exceed the brightness threshold across all color channels.
    // Used to identify specular highlights (glare) which can degrade SLAM accuracy.
    private fun glareMask(src: Bitmap, thr: Int): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val px = IntArray(out.width * out.height)
        out.getPixels(px, 0, out.width, 0, 0, out.width, out.height)
        for (i in px.indices) {
            val r = (px[i] shr 16) and 0xFF
            val g = (px[i] shr 8) and 0xFF
            val b = px[i] and 0xFF
            if (r > thr && g > thr && b > thr) {
                px[i] = (0xFF shl 24) or (0xFF shl 16) // pure red
            }
        }
        out.setPixels(px, 0, out.width, 0, 0, out.width, out.height)
        return out
    }

    @Composable
    fun GlareLabScreen() {
        var preset by remember { mutableIntStateOf(0) }
        var thr by remember { mutableFloatStateOf(225f) }
        var showDetections by remember { mutableStateOf(true) }
        val frames = listOf("hkreplay/f000.jpg", "hkreplay/f020.jpg",
            "hkreplay/f040.jpg", "hkreplay/f056.jpg")
        val src = remember(preset) { BitmapFactory.decodeStream(assets.open(frames[preset])) }
        val out = remember(preset, thr.toInt()) { glareMask(src, thr.toInt()) }
        Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Glare Lab", style = MaterialTheme.typography.headlineMedium)
            Text("Real colonoscopy frames, full of real glare. Slide the threshold and watch " +
                    "the detector decide what counts as a reflection.")
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        repeat(2) { j -> FilterChip(selected = preset == j, onClick = { preset = j },
                            label = { Text("frame ${j + 1}") }) }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        repeat(2) { j -> FilterChip(selected = preset == j + 2, onClick = { preset = j + 2 },
                            label = { Text("frame ${j + 3}") }) }
                    }
                }
                FilterChip(selected = showDetections, onClick = { showDetections = !showDetections },
                    label = { Text("show detections") })
            }
            Slider(value = thr, onValueChange = { thr = it }, valueRange = 180f..254f)
            Text("threshold: ${thr.toInt()}", style = MaterialTheme.typography.labelLarge)
            
            Image(
                bitmap = if (showDetections) out.asImageBitmap() else src.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxWidth().weight(1f)
            )
        }
    }
}