package com.example.asrmobile

/**
 * Whisper 推理引擎（JNI 封装）
 *
 * 通过 JNI 调用 whisper.cpp 进行语音识别。
 * 支持模型热切换：调用 [loadModel] 时会自动释放旧模型再加载新模型。
 *
 * JNI 层（whisper_jni.cpp）无需修改，因为所有加载都基于文件路径。
 */
class WhisperEngine {
    private var nativeHandle: Long = 0L
    private var currentModelPath: String? = null

    /** 当前是否已加载模型 */
    val isModelLoaded: Boolean get() = nativeHandle != 0L

    /** 当前已加载模型的路径，未加载时为 null */
    val loadedModelPath: String? get() = currentModelPath

    /**
     * 加载（或切换）模型
     *
     * - 如果已加载同一模型（路径相同），直接返回不重复加载
     * - 如果已加载不同模型，先释放旧模型再加载新模型
     */
    fun loadModel(modelPath: String) {
        if (nativeHandle != 0L && currentModelPath == modelPath) return

        // 释放旧模型后再加载新模型
        release()
        nativeHandle = initModel(modelPath)
        if (nativeHandle == 0L) {
            error("Native whisper.cpp backend is not configured yet, or the model could not be loaded.")
        }
        currentModelPath = modelPath
    }

    /**
     * 转写音频文件
     * @param wavPath 16kHz/16bit/mono WAV 文件路径
     * @return 识别文本
     */
    fun transcribe(wavPath: String): String {
        if (nativeHandle == 0L) {
            error("Model is not loaded. Load a model first.")
        }
        return transcribeFromFile(nativeHandle, wavPath)
    }

    /** 释放当前已加载的模型，释放后可以加载新模型 */
    fun release() {
        if (nativeHandle != 0L) {
            releaseModel(nativeHandle)
            nativeHandle = 0L
            currentModelPath = null
        }
    }

    protected fun finalize() {
        release()
    }

    private external fun initModel(modelPath: String): Long
    private external fun transcribeFromFile(handle: Long, wavPath: String): String
    private external fun releaseModel(handle: Long)

    companion object {
        init {
            System.loadLibrary("asr_mobile")
        }
    }
}
