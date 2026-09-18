package com.example.arguslite

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

/**
 * Holds a live study outside of any screen, so navigating away — to another tab, or to
 * another app entirely — never cancels or restarts the run. The UI just reads this state.
 */
object StudyRunner {
    var stage by mutableStateOf("")
    var frac by mutableFloatStateOf(0f)
    var running by mutableStateOf(false)
    var result by mutableStateOf<StudyEngine.StudyResult?>(null)
    var error by mutableStateOf<String?>(null)
    var sourceName by mutableStateOf("")
    var library by mutableStateOf<List<Pair<String, String>>>(emptyList())  // name -> url
    var libraryBusy by mutableStateOf(false)

    private var appCtx: Context? = null
    private var engine: StudyEngine? = null

    fun init(context: Context) {
        if (appCtx == null) appCtx = context.applicationContext
    }

    private fun engine(): StudyEngine {
        val ctx = appCtx!!
        return engine ?: StudyEngine(DepthEngine(ctx), PolypEngine(ctx)).also { engine = it }
    }

    fun startFromUri(context: Context, uri: Uri) {
        init(context)
        if (running) return
        running = true; result = null; error = null
        sourceName = displayName(context, uri)
        stage = "Reading video"; frac = 0f
        thread {
            try {
                val dest = File(appCtx!!.cacheDir, "study_pick.mp4")
                context.contentResolver.openInputStream(uri)!!.use { input ->
                    dest.outputStream().use { out -> input.copyTo(out) }
                }
                studyFile(dest)
            } catch (e: Exception) {
                error = "Could not open that video: ${e.message}"
                running = false
            }
        }
    }

    fun startFromUrl(rawUrl: String) {
        val url = rawUrl.trim()
        if (url.isEmpty() || running || appCtx == null) return
        running = true; result = null; error = null
        sourceName = url.substringAfterLast('/').ifEmpty { url }
        stage = "Fetching video"; frac = 0f
        thread {
            try {
                val dest = File(appCtx!!.cacheDir, "study_fetch.mp4")
                val conn = URL(url).openConnection() as HttpURLConnection
                conn.connectTimeout = 8000; conn.readTimeout = 20000
                val total = conn.contentLengthLong.coerceAtLeast(1L)
                conn.inputStream.use { input ->
                    dest.outputStream().use { out ->
                        val buf = ByteArray(64 * 1024)
                        var got = 0L
                        while (true) {
                            val nRead = input.read(buf)
                            if (nRead < 0) break
                            out.write(buf, 0, nRead)
                            got += nRead
                            frac = (got.toFloat() / total).coerceIn(0f, 1f)
                        }
                    }
                }
                studyFile(dest)
            } catch (e: Exception) {
                error = "Could not fetch that link: ${e.message}"
                running = false
            }
        }
    }

    /** Reads a plain directory listing (any simple file server) and lists its videos. */
    fun fetchLibrary(base: String) {
        val root = base.trim().removeSuffix("/")
        if (root.isEmpty() || libraryBusy) return
        libraryBusy = true
        thread {
            try {
                val html = URL("$root/").openStream().use { it.readBytes().decodeToString() }
                val found = Regex("href=\"([^\"]+\\.(?:mp4|webm))\"", RegexOption.IGNORE_CASE)
                    .findAll(html)
                    .map { it.groupValues[1] }
                    .distinct()
                    .map { name -> name.removePrefix("./") }
                    .map { name -> name to "$root/$name" }
                    .toList()
                library = found
                if (found.isEmpty()) error = "That library has no videos in it."
            } catch (e: Exception) {
                error = "Could not read that library: ${e.message}"
            }
            libraryBusy = false
        }
    }

    private fun studyFile(file: File) {
        try {
            val mmr = MediaMetadataRetriever()
            mmr.setDataSource(file.path)
            val durMs = mmr.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val want = 80
            val frames = ArrayList<Bitmap>(want)
            for (i in 0 until want) {
                stage = "Reading video"; frac = i.toFloat() / want
                val tUs = if (durMs > 0) i * durMs * 1000L / want else i * 200_000L
                val f = mmr.getFrameAtTime(tUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    ?: continue
                val w = 512
                val h = maxOf(1, w * f.height / f.width)
                val s = Bitmap.createScaledBitmap(f, w, h, true)
                if (s !== f) f.recycle()
                frames.add(s)
            }
            mmr.release()
            file.delete()
            if (frames.size < 10) {
                error = "Could not read enough frames from that video."
                running = false; return
            }
            result = engine().run(frames) { s, p -> stage = s; frac = p }
        } catch (e: Exception) {
            error = "The study failed partway through: ${e.message}"
        }
        running = false
    }

    private fun displayName(context: Context, uri: Uri): String {
        try {
            context.contentResolver.query(uri,
                arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) return c.getString(0) ?: "picked video"
            }
        } catch (_: Exception) { }
        return "picked video"
    }
}

/** The study launcher, embedded in the Studies screen: pick a file, paste a link, or
 *  browse a footage library served from any computer on the same network. */
@Composable
fun StudySourcePanel() {
    val context = LocalContext.current
    LaunchedEffect(Unit) { StudyRunner.init(context) }
    var link by rememberSaveable { mutableStateOf("") }
    var libBase by rememberSaveable { mutableStateOf("") }
    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()) { uri ->
        uri?.let { StudyRunner.startFromUri(context, it) }
    }
    Column(Modifier.fillMaxWidth().padding(top = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (StudyRunner.running) {
            Text(StudyRunner.stage, style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.primary)
            LinearProgressIndicator(progress = { StudyRunner.frac },
                Modifier.fillMaxWidth())
            Text("Running on this device. Other tabs — and other apps — are fine; the " +
                    "study keeps going.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            Text("Feed the pipeline footage it has never seen. Everything is computed " +
                    "during the run — nothing is prerecorded.",
                style = MaterialTheme.typography.bodySmall)
            Button(onClick = { picker.launch("video/*") }, Modifier.fillMaxWidth()) {
                Text("Choose a video from this phone")
            }
            OutlinedTextField(value = link, onValueChange = { link = it },
                label = { Text("Or paste a video link") }, singleLine = true,
                modifier = Modifier.fillMaxWidth())
            Button(onClick = { StudyRunner.startFromUrl(link) },
                enabled = link.isNotBlank(), modifier = Modifier.fillMaxWidth()) {
                Text("Fetch and study")
            }
            OutlinedTextField(value = libBase, onValueChange = { libBase = it },
                label = { Text("Or a footage library address") }, singleLine = true,
                modifier = Modifier.fillMaxWidth())
            Button(onClick = { StudyRunner.fetchLibrary(libBase) },
                enabled = libBase.isNotBlank() && !StudyRunner.libraryBusy,
                modifier = Modifier.fillMaxWidth()) {
                Text(if (StudyRunner.libraryBusy) "Looking…" else "List the library")
            }
            StudyRunner.library.forEach { (name, url) ->
                TextButton(onClick = { StudyRunner.startFromUrl(url) }) { Text(name) }
            }
            StudyRunner.result?.let {
                Text("Last study: ${StudyRunner.sourceName}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            StudyRunner.error?.let {
                Text(it, color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
