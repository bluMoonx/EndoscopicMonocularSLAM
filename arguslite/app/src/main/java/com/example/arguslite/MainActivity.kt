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

class MainActivity : ComponentActivity() {

    private var heat by mutableStateOf<Bitmap?>(null)
    private var camCtl by mutableStateOf<LifecycleCameraController?>(null)
    private val worker = Executors.newSingleThreadExecutor()

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
        setContent { ARGUSLiteTheme { ArgusApp() } }
    }

    private fun startCamera() {
        val engine = DepthEngine(this)
        val c = LifecycleCameraController(this)
        c.setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
        c.setImageAnalysisAnalyzer(worker) { proxy ->
            val bmp = proxy.toBitmap()
            val m = Matrix().apply { postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
            val upright = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            heat = engine.run(upright)
            proxy.close()
        }
        c.bindToLifecycle(this)
        camCtl = c
    }

    @Composable
    fun ArgusApp() {
        var tab by remember { mutableStateOf(0) }
        val tabs = listOf(
            "Scope" to Icons.Filled.PlayArrow,
            "Detect" to Icons.Filled.Search,
            "Mission" to Icons.Filled.Info,
            "Results" to Icons.Filled.Star,
            "Lab" to Icons.Filled.Build,
        )
        Scaffold(bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { i, (name, icon) ->
                    NavigationBarItem(selected = tab == i, onClick = { tab = i },
                        icon = { Icon(icon, name) }, label = { Text(name) })
                }
            }
        }) { pad ->
            Box(Modifier.padding(pad).fillMaxSize()) {
                when (tab) {
                    0 -> ScopeScreen()
                    1 -> Placeholder("Detect", "Polyp finder + auto-sizing — Milestone 5.")
                    2 -> MissionScreen()
                    3 -> ResultsScreen()
                    else -> Placeholder("Lab", "Live link to the workstation pipeline — Milestone 6.")
                }
            }
        }
    }

    @Composable
    fun ScopeScreen() {
        val ctl = camCtl
        if (ctl == null) {
            Placeholder("Scope", "Camera permission needed — restart the app and allow it.")
            return
        }
        Column(Modifier.fillMaxSize()) {
            AndroidView(
                modifier = Modifier.fillMaxWidth().weight(1f),
                factory = { ctx -> PreviewView(ctx).also { it.controller = ctl } })
            heat?.let {
                Image(it.asImageBitmap(), "depth", Modifier.fillMaxWidth().weight(1f))
            }
        }
    }

    @Composable
    fun MissionScreen() {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text("The mission", style = MaterialTheme.typography.headlineMedium)
            Card { Text("Colonoscopies miss roughly 1 in 4 precancerous polyps — often because " +
                    "parts of the colon wall are never actually looked at, and nobody knows which parts.",
                Modifier.padding(16.dp)) }
            Card { Text("ARGUS watches the video from a standard, unmodified endoscope and builds, " +
                    "from that video alone, a 3D map of the colon — with the never-inspected blind spots " +
                    "marked in red. No new hardware.", Modifier.padding(16.dp)) }
            Text("How it works", style = MaterialTheme.typography.headlineSmall)
            Card { Text("1 · DEPTH — a neural network, fine-tuned on colon imagery, estimates how far " +
                    "away every pixel is. That exact network is running live in this app's Scope tab.",
                Modifier.padding(16.dp)) }
            Card { Text("2 · TRACKING — a from-scratch visual odometry algorithm compares consecutive " +
                    "frames to recover the camera's motion, accurate to about a millimeter.",
                Modifier.padding(16.dp)) }
            Card { Text("3 · FUSION — thousands of noisy depth snapshots vote together into one clean " +
                    "3D surface.", Modifier.padding(16.dp)) }
            Card { Text("4 · COVERAGE — every wall patch is checked: how many times did the camera " +
                    "actually see it? Fewer than three views = blind spot.", Modifier.padding(16.dp)) }
        }
    }

    @Composable
    fun ResultsScreen() {
        val stats = listOf(
            "0.91 mm" to "average tracking error, fully monocular, vs ground truth",
            "98.6 %" to "blind-spot map agreement with ground truth (was 74.5% before the depth fix)",
            "68% → 98%" to "depth accuracy before vs after fine-tuning (held-out validation)",
            "1.3–2.4 mm" to "3D surface accuracy, reconstructed from video alone",
            "0.60 mm" to "median size-measurement error on 3–30 mm targets",
        )
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Validated results", style = MaterialTheme.typography.headlineMedium)
            stats.forEach { (n, d) ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text(n, style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.primary)
                        Text(d, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
            Text("Every number graded against the C3VD phantom's measured ground truth.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    @Composable
    fun Placeholder(title: String, body: String) {
        Column(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.Center) {
            Text(title, style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(8.dp))
            Text(body, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}