package com.ai.assistance.operit.api.chat.llmprovider

import com.ai.assistance.operit.data.model.ApiProviderType
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MediaWireCapabilityTest {

    private fun format(
        providerType: ApiProviderType?,
        modelName: String?,
        endpoint: String = "",
    ): MediaWireCapability.VideoWireFormat =
        MediaWireCapability.resolveVideoWireFormat(providerType, modelName, endpoint)

    @Test
    fun deepseekChatModelNeverEmitsVideoUrl() {
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(ApiProviderType.DEEPSEEK, "deepseek-v4-flash", "https://api.deepseek.com/v1/chat/completions")
        )
    }

    @Test
    fun deepseekRejectedEvenWhenProviderMisconfigured() {
        // 用户把 DeepSeek 端点填成 OPENAI_GENERIC 也不能绕过
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(
                ApiProviderType.OPENAI_GENERIC,
                "deepseek-v4-flash",
                "https://api.deepseek.com/v1/chat/completions"
            )
        )
    }

    @Test
    fun deepseekRejectedThroughAggregatorModelId() {
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(ApiProviderType.OPENROUTER, "deepseek/deepseek-v4-flash", "https://openrouter.ai/api/v1")
        )
    }

    @Test
    fun openRouterSameProviderDifferentModelsResolveDifferently() {
        // 模型级能力优先：同一个聚合平台里，视觉模型放行、纯文本模型拒绝
        assertEquals(
            MediaWireCapability.VideoWireFormat.VIDEO_URL,
            format(ApiProviderType.OPENROUTER, "qwen/qwen3-vl-235b-a22b-instruct", "https://openrouter.ai/api/v1")
        )
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(ApiProviderType.OPENROUTER, "meta-llama/llama-3.3-70b-instruct", "https://openrouter.ai/api/v1")
        )
    }

    @Test
    fun ollamaTagIsNormalised() {
        assertEquals(
            MediaWireCapability.VideoWireFormat.VIDEO_URL,
            format(ApiProviderType.ALIYUN, "qwen2.5-vl:7b", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
    }

    @Test
    fun unknownModelDefaultsToNoVideo() {
        assertEquals(MediaWireCapability.VideoWireFormat.NONE, format(ApiProviderType.OPENAI, "gpt-4o"))
        assertEquals(MediaWireCapability.VideoWireFormat.NONE, format(ApiProviderType.OPENAI_GENERIC, "my-private-model"))
        assertEquals(MediaWireCapability.VideoWireFormat.NONE, format(ApiProviderType.OPENROUTER, ""))
        assertEquals(MediaWireCapability.VideoWireFormat.NONE, format(null, "qwen3-vl-plus"))
    }

    @Test
    fun videoCapableModelOnUnlistedProviderStaysOff() {
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(ApiProviderType.XAI, "qwen3-vl-plus", "https://api.x.ai/v1")
        )
    }

    @Test
    fun deepseekProviderBeatsVideoCapableModelName() {
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(ApiProviderType.DEEPSEEK, "qwen3-vl-plus", "https://api.deepseek.com/v1")
        )
    }

    @Test
    fun deepseekHostBlockedEvenWithCustomProviderAndVisionModelName() {
        assertEquals(
            MediaWireCapability.VideoWireFormat.NONE,
            format(
                ApiProviderType.OPENAI_GENERIC,
                "qwen3-vl-plus",
                "https://api.deepseek.com/v1/chat/completions"
            )
        )
    }

    @Test
    fun noticeStatesVideoWasNotAnalyzed() {
        val notice = MediaWireCapability.unsupportedVideoNotice("deepseek-v4-flash", ApiProviderType.DEEPSEEK)
        assertTrue(notice.contains("NOT analyzed"))
        assertTrue(notice.contains("deepseek-v4-flash"))
        assertFalse(notice.contains("described the video"))
    }

    @Test
    fun responsesVideoPartsAreStrippedAndOtherTypesSurvive() {
        val request = responsesRequest(
            JSONArray()
                .put(videoPart("data:video/mp4;base64,ZmFrZQ=="))
                .put(imagePart())
                .put(JSONObject().apply { put("type", "input_file"); put("filename", "a.pdf"); put("file_data", "data:application/pdf;base64,ZmFrZQ==") })
                .put(JSONObject().apply { put("type", "input_text"); put("text", "这个视频讲了什么") })
        )

        MediaWireCapability.stripUnsupportedResponsesVideoParts(
            request = request,
            allowVideo = false,
            modelName = "deepseek-v4-flash",
            providerType = ApiProviderType.DEEPSEEK,
        )

        val content = contentOf(request)
        val types = (0 until content.length()).map { content.getJSONObject(it).getString("type") }
        assertEquals(listOf("input_image", "input_file", "input_text"), types)
        assertFalse(content.toString().contains("input_video"))
        assertFalse(content.toString().contains("video_url"))
        assertEquals("这个视频讲了什么", content.getJSONObject(2).getString("text"))
    }

    @Test
    fun strippedVideoLeavesExplicitPlaceholderInsteadOfBlankContent() {
        val request = responsesRequest(JSONArray().put(videoPart("data:video/mp4;base64,ZmFrZQ==")))

        MediaWireCapability.stripUnsupportedResponsesVideoParts(
            request = request,
            allowVideo = false,
            modelName = "deepseek-v4-flash",
            providerType = ApiProviderType.DEEPSEEK,
        )

        val content = contentOf(request)
        assertEquals(1, content.length())
        assertEquals("input_text", content.getJSONObject(0).getString("type"))
        assertTrue(content.getJSONObject(0).getString("text").contains("NOT analyzed"))
    }

    @Test
    fun allowedProviderKeepsVideoPartsUntouched() {
        val request = responsesRequest(JSONArray().put(videoPart("data:video/mp4;base64,ZmFrZQ==")))

        MediaWireCapability.stripUnsupportedResponsesVideoParts(
            request = request,
            allowVideo = true,
            modelName = "qwen3-vl-plus",
            providerType = ApiProviderType.ALIYUN,
        )

        val content = contentOf(request)
        assertEquals(1, content.length())
        assertEquals("input_video", content.getJSONObject(0).getString("type"))
    }

    private fun videoPart(url: String): JSONObject =
        JSONObject().apply {
            put("type", "input_video")
            put("video_url", url)
        }

    private fun imagePart(): JSONObject =
        JSONObject().apply {
            put("type", "input_image")
            put("image_url", "data:image/jpeg;base64,ZmFrZQ==")
        }

    private fun responsesRequest(content: JSONArray): JSONObject =
        JSONObject().apply {
            put("model", "deepseek-v4-flash")
            put(
                "input",
                JSONArray().put(
                    JSONObject().apply {
                        put("role", "user")
                        put("content", content)
                    }
                )
            )
        }

    private fun contentOf(request: JSONObject): JSONArray =
        request.getJSONArray("input").getJSONObject(0).getJSONArray("content")
}