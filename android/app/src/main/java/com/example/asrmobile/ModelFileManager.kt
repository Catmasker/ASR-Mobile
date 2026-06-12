package com.example.asrmobile

import android.content.Context
import android.net.Uri
import java.io.File

/**
 * 模型文件管理器 — 外部文件导入
 *
 * 负责从系统文件选择器等外部来源导入模型文件到 app 内部存储。
 *
 * 内置模型（打包在 APK assets 中的模型）的查询与部署请使用 [ModelRepository]。
 */
class ModelFileManager(private val context: Context) {

    /**
     * 将用户通过系统文件选择器选中的模型文件拷贝到 app 内部存储
     * @param uri 从 [android.content.Intent.ACTION_OPEN_DOCUMENT] 获取到的 content URI
     * @return 拷贝后的 File 对象
     */
    fun copyModelUriToAppStorage(uri: Uri): File {
        val modelsDir = File(context.filesDir, "models").apply { mkdirs() }
        val displayName = queryDisplayName(uri) ?: "selected-model.bin"
        val safeName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val outputFile = File(modelsDir, safeName)

        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Cannot open selected model file." }
            outputFile.outputStream().use { output -> input.copyTo(output) }
        }
        return outputFile
    }

    private fun queryDisplayName(uri: Uri): String? {
        val projection = arrayOf(android.provider.OpenableColumns.DISPLAY_NAME)
        context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (index >= 0) return cursor.getString(index)
            }
        }
        return uri.lastPathSegment
    }
}
