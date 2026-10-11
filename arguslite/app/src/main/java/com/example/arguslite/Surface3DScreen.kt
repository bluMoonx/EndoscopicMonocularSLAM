package com.example.arguslite

import android.annotation.SuppressLint
import android.webkit.WebView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

/** The showcase reconstructions as REAL lit surfaces, rendered by the phone's GPU
 *  through a WebView (three.js, fully offline assets). The live-study map stays the
 *  coarse real-time coverage instrument; these are the workstation's 0.6 mm fusions —
 *  the same files Studio station 08 serves, bundled into the APK.
 *  Toggle tells the depth story: fine-tuned vs off-the-shelf depth, same video. */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun Surface3DScreen() {
    var page by remember { mutableStateOf("surface_ft.html") }
    var web by remember { mutableStateOf<WebView?>(null) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(16.dp, 16.dp, 16.dp, 8.dp)) {
            Text("3D Surfaces", style = MaterialTheme.typography.headlineMedium)
            Text("The full-resolution reconstruction (0.6 mm voxels, built from video " +
                    "alone on the workstation), rendered live by this phone's GPU. " +
                    "Drag to orbit, pinch to zoom. The toggle is the project's core " +
                    "discovery: the same pipeline with off-the-shelf depth versus " +
                    "colon-fine-tuned depth.",
                style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = page == "surface_ft.html",
                    onClick = { page = "surface_ft.html"
                        web?.loadUrl("file:///android_asset/surface_ft.html") },
                    label = { Text("Fine-tuned depth") })
                FilterChip(selected = page == "surface_mono.html",
                    onClick = { page = "surface_mono.html"
                        web?.loadUrl("file:///android_asset/surface_mono.html") },
                    label = { Text("Off-the-shelf (the before)") })
            }
        }
        AndroidView(
            factory = { ctx ->
                WebView(ctx).apply {
                    settings.javaScriptEnabled = true
                    settings.allowFileAccess = true
                    setBackgroundColor(android.graphics.Color.TRANSPARENT)
                    loadUrl("file:///android_asset/surface_ft.html")
                    web = this
                }
            },
            modifier = Modifier.fillMaxSize()
        )
    }
}
