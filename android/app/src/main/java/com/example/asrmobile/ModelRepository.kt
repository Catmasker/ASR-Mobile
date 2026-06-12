package com.example.asrmobile

import android.content.Context
import java.io.File

/**
 * 内置模型元数据
 *
 * @property fileName    assets/models/ 中的文件名，如 "ggml-tiny.bin"
 * @property displayName UI 上显示的名称，如 "Whisper Tiny"
 * @property description 简短描述（语言、速度、质量等）
 * @property estimatedSizeMB 模型文件预估大小（MB）
 */
data class BundledModel(
    val fileName: String,
    val displayName: String,
    val description: String,
    val estimatedSizeMB: Long,
)

/**
 * 模型仓库（Model Repository）
 *
 * ── 职责 ──
 * 后端模型管理接口，统一管理所有 App 内置的 ASR 模型：
 * 1. 注册：所有内置模型在 [AVAILABLE_MODELS] 中登记
 * 2. 查询：检查模型是否存在于 APK assets 中
 * 3. 部署：将模型从 assets 拷贝到运行时目录供 WhisperEngine 加载
 *
 * ── 使用方式 ──
 *   val repo = ModelRepository(context)
 *   val models = repo.getBundledModels()         // 获取所有注册的模型
 *   val path  = repo.deployModel(models[0])      // 部署第一个模型
 *
 * ── 添加新模型 ──
 * 1. 将 .bin 模型文件放入 android/app/src/main/assets/models/
 * 2. 在 [AVAILABLE_MODELS] 中添加一条 BundledModel(...) 记录
 * 3. 重新编译 APK，新模型自动出现在内置模型列表中
 */
class ModelRepository(private val context: Context) {

    companion object {
        /**
         * 模型仓库注册表
         *
         * 所有 App 内置模型在此登记。这是添加新模型的唯一入口。
         * 按推荐程度排序，最推荐的在最前。
         */
        val AVAILABLE_MODELS: List<BundledModel> = listOf(
            BundledModel(
                fileName = "ggml-tiny.bin",
                displayName = "Whisper Tiny",
                description = "多语言, 速度最快, 适合手机端演示 (~77MB)",
                estimatedSizeMB = 77,
            ),
            // ──────────────────────────────────────────────────────────
            // 添加新模型示例（取消注释后放入对应 .bin 文件即可）：
            //
            // BundledModel(
            //     fileName = "ggml-base.bin",
            //     displayName = "Whisper Base",
            //     description = "多语言, 质量较好, 速度适中 (~150MB)",
            //     estimatedSizeMB = 150,
            // ),
            // BundledModel(
            //     fileName = "ggml-small.bin",
            //     displayName = "Whisper Small",
            //     description = "多语言, 高质量, 需较高配置 (~460MB)",
            //     estimatedSizeMB = 460,
            // ),
            // ──────────────────────────────────────────────────────────
        )

        private const val MODELS_ASSET_DIR = "models"
    }

    /** 获取所有已注册的内置模型列表 */
    fun getBundledModels(): List<BundledModel> = AVAILABLE_MODELS

    /** 检查模型文件是否存在于 APK assets 中 */
    fun hasModel(model: BundledModel): Boolean = runCatching {
        context.assets.open("$MODELS_ASSET_DIR/${model.fileName}").use { true }
    }.getOrDefault(false)

    /** 获取模型在运行时存储中的路径（如果已部署过且文件完整） */
    fun getModelPath(model: BundledModel): File? {
        val file = File(context.filesDir, "models/${model.fileName}")
        return if (file.exists() && file.length() > 0L) file else null
    }

    /**
     * 部署模型：将模型从 assets 拷贝到 app 内部存储
     *
     * - 如果已经部署过且文件完整，直接返回已有文件
     * - 否则从 assets 拷贝到 [context.filesDir]/models/
     *
     * @return 部署完成后的 File 对象
     */
    fun deployModel(model: BundledModel): File {
        val modelsDir = File(context.filesDir, "models").apply { mkdirs() }
        val outputFile = File(modelsDir, model.fileName)
        if (outputFile.exists() && outputFile.length() > 0L) return outputFile

        context.assets.open("$MODELS_ASSET_DIR/${model.fileName}").use { input ->
            outputFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outputFile
    }
}
