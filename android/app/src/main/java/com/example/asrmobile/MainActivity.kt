package com.example.asrmobile

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File

class MainActivity : AppCompatActivity() {
    private val modelFileManager by lazy { ModelFileManager(this) }
    private val modelRepository by lazy { ModelRepository(this) }
    private val whisperEngine by lazy { WhisperEngine() }
    private val audioRecorder by lazy { AudioRecorder(this) }
    private val benchmarkRunner by lazy { BenchmarkRunner(this, whisperEngine) }

    private lateinit var statusText: TextView
    private lateinit var transcriptText: TextView
    private lateinit var metricsText: TextView

    private var selectedModelPath: String? = null
    private var latestRecording: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        requestMicrophonePermissionIfNeeded()
        updateStatus("Ready. Select a built-in model or pick a model file.")
    }

    private fun buildUi() {
        statusText = TextView(this).withPadding()
        transcriptText = TextView(this).withPadding()
        metricsText = TextView(this).withPadding()

        // ── 标题 ──
        val title = TextView(this).apply {
            text = "ASR Mobile"
            textSize = 24f
        }

        // ── 外部文件选择 ──
        val selectFileButton = Button(this).apply {
            text = "Select model file from storage"
            setOnClickListener { selectModelFile() }
        }

        // ── 内置模型仓库列表 ──
        val builtinHeader = TextView(this).apply {
            text = "── Built-in Models ──"
            textSize = 16f
            setPadding(0, 16, 0, 8)
        }

        val modelListLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        populateModelList(modelListLayout)

        // ── 功能按钮 ──
        val loadModelButton = Button(this).apply {
            text = "Load model"
            setOnClickListener { loadSelectedModel() }
        }
        val recordButton = Button(this).apply {
            text = "Record 10 seconds"
            setOnClickListener { recordShortClip() }
        }
        val transcribeButton = Button(this).apply {
            text = "Transcribe latest recording"
            setOnClickListener { transcribeLatestRecording() }
        }
        val benchmarkButton = Button(this).apply {
            text = "Run benchmark"
            setOnClickListener { runBenchmark() }
        }

        // ── 布局组装 ──
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            addView(title)
            addView(statusText)
            addView(selectFileButton)
            addView(builtinHeader)
            addView(modelListLayout)
            addView(loadModelButton)
            addView(recordButton)
            addView(transcribeButton)
            addView(benchmarkButton)
            addView(TextView(context).apply { text = "Transcript"; textSize = 18f })
            addView(transcriptText)
            addView(TextView(context).apply { text = "Metrics"; textSize = 18f })
            addView(metricsText)
        }

        setContentView(ScrollView(this).apply { addView(content) })
    }

    /**
     * 动态填充内置模型按钮列表
     * 每个按钮点击后将该模型部署到运行时存储并设置为待加载模型
     */
    private fun populateModelList(container: LinearLayout) {
        val models = modelRepository.getBundledModels()
        if (models.isEmpty()) {
            container.addView(TextView(context).apply {
                text = "(No built-in models registered)"
                setPadding(16, 8, 16, 8)
            })
            return
        }

        for (model in models) {
            val isAvailable = modelRepository.hasModel(model)
            val button = Button(this).apply {
                val status = if (isAvailable) "" else " [FILE NOT FOUND in assets/]"
                text = "${model.displayName}  (${model.estimatedSizeMB}MB)$status"
                isEnabled = isAvailable
                setOnClickListener { selectBundledModel(model) }
            }
            container.addView(button)

            // 描述文字
            if (model.description.isNotBlank()) {
                container.addView(TextView(context).apply {
                    text = model.description
                    textSize = 12f
                    setPadding(24, 0, 0, 8)
                })
            }
        }
    }

    // ══════════════════════════════════════════
    //  模型选择
    // ══════════════════════════════════════════

    /** 选择内置模型：部署到运行时存储，标记为待加载 */
    private fun selectBundledModel(model: BundledModel) {
        updateStatus("Preparing ${model.displayName}...")
        selectedModelPath = null
        Thread {
            val result = runCatching { modelRepository.deployModel(model).absolutePath }
            runOnUiThread {
                result.onSuccess { path ->
                    selectedModelPath = path
                    updateStatus("Selected built-in model: ${model.displayName}")
                }.onFailure {
                    updateStatus("Failed to prepare model: ${it.message}")
                }
            }
        }.start()
    }

    /** 从系统文件选择器选取外部模型文件 */
    private fun selectModelFile() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }
        startActivityForResult(intent, REQUEST_MODEL_FILE)
    }

    @Deprecated("Deprecated in AndroidX Activity Result API, but kept simple for this teaching scaffold.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_MODEL_FILE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
                val path = modelFileManager.copyModelUriToAppStorage(uri).absolutePath
                selectedModelPath = path
                updateStatus("Selected external model file: $path")
            }
        }
    }

    // ══════════════════════════════════════════
    //  模型加载 / 推理 / 评测
    // ══════════════════════════════════════════

    /** 加载当前选中的模型到 WhisperEngine（JNI -> whisper.cpp） */
    private fun loadSelectedModel() {
        val path = selectedModelPath
        if (path == null) {
            updateStatus("No model selected. Pick a built-in model or a model file first.")
            return
        }

        updateStatus("Loading model...")
        Thread {
            val result = runCatching { whisperEngine.loadModel(path) }
            runOnUiThread {
                result.onSuccess { updateStatus("Model loaded successfully.") }
                    .onFailure { updateStatus("Model load failed: ${it.message}") }
            }
        }.start()
    }

    private fun recordShortClip() {
        if (!hasMicrophonePermission()) {
            requestMicrophonePermissionIfNeeded()
            return
        }

        updateStatus("Recording 10 seconds...")
        Thread {
            val recording = audioRecorder.recordBlocking(seconds = 10)
            latestRecording = recording
            runOnUiThread { updateStatus("Recording saved: ${recording.absolutePath}") }
        }.start()
    }

    private fun transcribeLatestRecording() {
        val recording = latestRecording
        if (recording == null) {
            updateStatus("No recording yet.")
            return
        }

        updateStatus("Transcribing ${recording.name}...")
        Thread {
            val result = runCatching { whisperEngine.transcribe(recording.absolutePath) }
            runOnUiThread {
                result.onSuccess { transcriptText.text = it }
                    .onFailure { transcriptText.text = "Transcription failed: ${it.message}" }
                updateStatus("Transcription finished.")
            }
        }.start()
    }

    private fun runBenchmark() {
        val recording = latestRecording
        if (recording == null) {
            updateStatus("Record a clip before benchmarking.")
            return
        }

        Thread {
            val result = benchmarkRunner.benchmark(recording, selectedModelPath)
            runOnUiThread { metricsText.text = result.toDisplayText() }
        }.start()
    }

    // ══════════════════════════════════════════
    //  权限
    // ══════════════════════════════════════════

    private fun requestMicrophonePermissionIfNeeded() {
        if (!hasMicrophonePermission()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO
            )
        }
    }

    private fun hasMicrophonePermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun updateStatus(message: String) {
        statusText.text = message
    }

    private fun TextView.withPadding(): TextView = apply { setPadding(0, 12, 0, 12) }

    companion object {
        private const val REQUEST_RECORD_AUDIO = 1001
        private const val REQUEST_MODEL_FILE = 1002
    }
}
