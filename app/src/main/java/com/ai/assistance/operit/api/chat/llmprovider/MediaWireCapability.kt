package com.ai.assistance.operit.api.chat.llmprovider

import com.ai.assistance.operit.data.model.ApiProviderType
import org.json.JSONArray
import org.json.JSONObject

/**
 * 决定某个 Provider + 具体模型可以把哪些多模态内容块放到请求里。
 *
 * 判定优先级：
 * 1. 具体模型能力表（模型名优先，支持聚合平台的 "vendor/model" 与 "model:tag" 写法）；
 * 2. Provider / 端点白名单；
 * 3. 无法识别时一律不发视频块（安全默认，绝不猜测）。
 *
 * 该对象只影响请求构建阶段，不会改动任何已经持久化的历史消息或附件。
 */
object MediaWireCapability {

    enum class VideoWireFormat {
        NONE,
        VIDEO_URL,
    }

    /** 明确不接受 video 内容块的模型族（DeepSeek 官方仅有 text/image_url/file）。 */
    private val VIDEO_REJECTING_MODEL_MARKERS = listOf("deepseek")

    /** 指向这些端点的请求一律不发视频块，即使 Provider 类型或模型名被填错。 */
    private val VIDEO_REJECTING_ENDPOINT_MARKERS = listOf("deepseek.com")

    /**
     * 已确认接受 video 内容块的模型族。
     * 只放能确认的；未列出的模型保持默认拒绝。
     */
    private val VIDEO_CAPABLE_MODEL_PATTERNS = listOf(
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "qwen-omni",
        "doubao-1.5-vision",
        "doubao-1-5-vision",
        "doubao-vision",
        "doubao-seed",
    )

    /** 允许发送 video_url 的 Provider；聚合平台也在此列，但必须同时命中具体模型能力表。 */
    private val PROVIDERS_ALLOWING_VIDEO_URL = setOf(
        ApiProviderType.ALIYUN,
        ApiProviderType.DOUBAO,
        ApiProviderType.SILICONFLOW,
        ApiProviderType.IFLOW,
        ApiProviderType.FOUR_ROUTER,
        ApiProviderType.NOUS_PORTAL,
        ApiProviderType.OPENROUTER,
        ApiProviderType.OPENAI_GENERIC,
        ApiProviderType.OTHER,
    )

    private val VIDEO_CAPABLE_ENDPOINT_MARKERS = listOf(
        "dashscope",
        "aliyuncs.com",
        "volces.com",
        "volcengine",
        "openrouter.ai",
        "siliconflow",
        "api.iflow.cn",
    )

    fun resolveVideoWireFormat(
        providerType: ApiProviderType?,
        modelName: String?,
        apiEndpoint: String?,
    ): VideoWireFormat {
        if (providerType == ApiProviderType.DEEPSEEK) return VideoWireFormat.NONE

        val endpoint = (apiEndpoint ?: "").lowercase()
        // 自定义 Provider 指向 DeepSeek 端点时同样硬禁，避免模型名乱填绕过
        if (VIDEO_REJECTING_ENDPOINT_MARKERS.any { endpoint.contains(it) }) {
            return VideoWireFormat.NONE
        }

        val fullName = (modelName ?: "").trim().lowercase()
        if (fullName.isEmpty()) return VideoWireFormat.NONE

        // openrouter/qwen/qwen3-vl-235b、ollama 的 model:tag 都归一化后再判断
        val shortName = fullName.substringAfterLast('/').substringBefore(':').trim()
        if (isVideoRejectingModel(fullName) || isVideoRejectingModel(shortName)) {
            return VideoWireFormat.NONE
        }

        val modelSupportsVideo = VIDEO_CAPABLE_MODEL_PATTERNS.any { pattern ->
            shortName.startsWith(pattern) || fullName.contains(pattern)
        }
        if (!modelSupportsVideo) return VideoWireFormat.NONE

        val providerAllowsVideo = providerType in PROVIDERS_ALLOWING_VIDEO_URL ||
            VIDEO_CAPABLE_ENDPOINT_MARKERS.any { endpoint.contains(it) }
        return if (providerAllowsVideo) VideoWireFormat.VIDEO_URL else VideoWireFormat.NONE
    }

    fun isVideoRejectingModel(modelName: String?): Boolean {
        val normalized = (modelName ?: "").trim().lowercase()
        if (normalized.isEmpty()) return false
        return VIDEO_REJECTING_MODEL_MARKERS.any { normalized.contains(it) }
    }

    /**
     * 发送不出去时给模型看的文字占位。必须明确说明视频没有被分析，
     * 不能伪装成已经看过画面。
     */
    fun unsupportedVideoNotice(modelName: String?, providerType: ApiProviderType?): String {
        val model = (modelName ?: "").ifBlank { "unknown" }
        val provider = providerType?.name ?: "unknown"
        return "[video not sent] The current model \"$model\" ($provider) does not accept video input. " +
            "The attached video was NOT analyzed and no frame was inspected. " +
            "Treat it only as a user attachment reference; ask the user or use a video-capable model instead."
    }

    /**
     * Responses API 侧兜底：把 input_video / video_url 内容块剥掉。
     * 只在构建请求时使用，历史数据保持原样。
     */
    fun stripUnsupportedResponsesVideoParts(
        request: JSONObject,
        allowVideo: Boolean,
        modelName: String?,
        providerType: ApiProviderType?,
    ): JSONObject {
        if (allowVideo) return request
        val input = request.optJSONArray("input") ?: return request

        for (i in 0 until input.length()) {
            val item = input.optJSONObject(i) ?: continue
            val content = item.optJSONArray("content") ?: continue
            val kept = JSONArray()
            var removed = false
            for (j in 0 until content.length()) {
                val part = content.optJSONObject(j) ?: continue
                val type = part.optString("type", "")
                if (type == "input_video" || type == "video_url") {
                    removed = true
                    continue
                }
                kept.put(part)
            }
            if (!removed) continue
            if (kept.length() == 0) {
                kept.put(
                    JSONObject().apply {
                        put("type", "input_text")
                        put("text", unsupportedVideoNotice(modelName, providerType))
                    }
                )
            }
            item.put("content", kept)
        }
        return request
    }
}
