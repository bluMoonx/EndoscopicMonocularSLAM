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
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private var heat by mutableStateOf<Bitmap?>(null)
    private val worker = Executors.newSingleThreadExecutor()

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) showCamera() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) showCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun showCamera() {
        val engine = DepthEngine(this)
        val controller = LifecycleCameraController(this)
        controller.setEnabledUseCases(
            CameraController.IMAGE_ANALYSIS or CameraController.IMAGE_CAPTURE)
        controller.setImageAnalysisAnalyzer(worker) { proxy ->
            val bmp = proxy.toBitmap()
            val m = Matrix().apply { postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
            val upright = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            heat = engine.run(upright)
            proxy.close()
        }
        controller.bindToLifecycle(this)
        setContent {
            Column(Modifier.fillMaxSize()) {
                AndroidView(
                    modifier = Modifier.fillMaxWidth().weight(1f),
                    factory = { ctx -> PreviewView(ctx).also { it.controller = controller } }
                )
                heat?.let {
                    Image(it.asImageBitmap(), "depth",
                        Modifier.fillMaxWidth().weight(1f))
                }
            }
        }
    }
}