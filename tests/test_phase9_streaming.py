import json
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from ai.provider import AIProvider, GeminiProvider, Message, ProviderError
from config import Config
from core.assistant import Assistant
from core.confirmation import AutoApproveHandler, ConfirmationHandler
from core.tool_router import ToolRouter
from main import run_voice_mode, _speak_streamed_response
from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry
from voice.stream_buffer import SentenceBuffer
from voice.text_to_speech import TextToSpeech


# ============================================================================
# 1. Provider Streaming Tests
# ============================================================================

class TestProviderStreaming:
    def test_generate_stream_yields_incremental_chunks(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.iter_lines.return_value = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"world!"}]}}]}',
            "data: [DONE]",
        ]
        session.post.return_value = response
        provider = GeminiProvider("test-key", "gemini-3.5-flash-lite", session=session)

        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["Hello ", "world!"]
        assert session.post.call_args.kwargs["stream"] is True

    def test_generate_stream_sends_api_key_in_header(self):
        secret = "SUPER_SECRET_KEY_12345"
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.iter_lines.return_value = ["data: [DONE]"]
        session.post.return_value = response
        provider = GeminiProvider(secret, "test-model", session=session)

        assert list(provider.generate_stream([{"role": "user", "content": "hi"}])) == []

        args, kwargs = session.post.call_args
        assert secret not in args[0]
        assert kwargs["headers"]["x-goog-api-key"] == secret
        response.close.assert_called_once()

    def test_generate_stream_ignores_empty_and_malformed_chunks(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.iter_lines.return_value = [
            "",
            "event: ping",
            "data: ",
            "data: not a json string",
            'data: {"candidates":[{"content":{"parts":[{"text":""}]}}]}',
            'data: {"candidates":[]}',
            'data: {"invalid_key": 123}',
            'data: {"candidates":[{"content":{"parts":[{"text":"Valid output."}]}}]}',
            "data: [DONE]",
        ]
        session.post.return_value = response
        provider = GeminiProvider("test-key", "gemini-3.5-flash-lite", session=session)

        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["Valid output."]

    def test_generate_stream_http_errors_raise_provider_error(self):
        status_codes = [400, 401, 403, 404, 429, 500, 503]
        for code in status_codes:
            session = MagicMock()
            response = MagicMock()
            response.status_code = code
            response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=response)
            session.post.return_value = response
            provider = GeminiProvider("test-key", "gemini-3.5-flash-lite", session=session)

            with pytest.raises(ProviderError):
                list(provider.generate_stream([{"role": "user", "content": "hi"}]))

    def test_generate_stream_timeout_and_connection_error_raise_provider_error(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("Read timeout")
        provider = GeminiProvider("test-key", "gemini-3.5-flash-lite", session=session)

        with pytest.raises(ProviderError, match="timed out"):
            list(provider.generate_stream([{"role": "user", "content": "hi"}]))

        session.post.side_effect = requests.exceptions.ConnectionError("Failed to connect")
        with pytest.raises(ProviderError, match="connect"):
            list(provider.generate_stream([{"role": "user", "content": "hi"}]))

    def test_generate_stream_respects_cancellation_event(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        cancel_event = threading.Event()

        def line_generator(*args, **kwargs):
            yield 'data: {"candidates":[{"content":{"parts":[{"text":"Chunk 1. "}]}}]}'
            cancel_event.set()
            yield 'data: {"candidates":[{"content":{"parts":[{"text":"Chunk 2."}]}}]}'

        response.iter_lines.side_effect = line_generator
        session.post.return_value = response
        provider = GeminiProvider("test-key", "gemini-3.5-flash-lite", session=session)

        chunks = list(provider.generate_stream([{"role": "user", "content": "hi"}], cancel_event=cancel_event))
        assert chunks == ["Chunk 1. "]
        response.close.assert_called()


# ============================================================================
# 2. Sentence Buffering Tests
# ============================================================================

class TestSentenceBuffering:
    def test_sentence_buffer_accumulates_until_min_chars_and_punctuation(self):
        buffer = SentenceBuffer(min_chars=20, max_chars=100)
        assert buffer.feed("Hello. ") == []  # length 7 < 20
        assert buffer.feed("How are you today? ") == ["Hello. How are you today?"]
        assert buffer.flush() == []

    def test_sentence_buffer_handles_multiple_sentences_in_single_chunk(self):
        buffer = SentenceBuffer(min_chars=10, max_chars=100)
        result = buffer.feed("Sentence one. Sentence two. Sentence three.")
        assert result == ["Sentence one.", "Sentence two.", "Sentence three."]
        assert buffer.flush() == []

    def test_sentence_buffer_flushes_at_max_chars_word_boundary_without_punctuation(self):
        buffer = SentenceBuffer(min_chars=10, max_chars=25)
        # Feed 35 chars without punctuation
        result = buffer.feed("The quick brown fox jumps over dog")
        assert len(result) >= 1
        assert "The quick brown" in result[0]
        remaining = buffer.flush()
        assert len(remaining) >= 1
        assert "dog" in remaining[0]

    def test_sentence_buffer_empty_and_whitespace_input(self):
        buffer = SentenceBuffer(min_chars=10, max_chars=50)
        assert buffer.feed("") == []
        assert buffer.feed(None) == []  # type: ignore
        assert buffer.flush() == []

    def test_sentence_buffer_clear_discards_pending(self):
        buffer = SentenceBuffer(min_chars=20, max_chars=50)
        buffer.feed("Unfinished pending segment")
        buffer.clear()
        assert buffer.flush() == []


# ============================================================================
# 3. TTS Streaming Queue & Ordering Tests
# ============================================================================

class TestTTSStreamingQueue:
    def test_tts_speak_async_queues_and_preserves_order(self):
        spoken_items = []

        tts = TextToSpeech()
        tts._engine = MagicMock()

        def fake_speak(text):
            spoken_items.append(text)

        tts.speak = fake_speak

        tts.speak_async("Sentence A")
        tts.speak_async("Sentence B")
        tts.speak_async("Sentence C")

        # Wait for worker thread to process items
        time.sleep(0.1)
        assert spoken_items == ["Sentence A", "Sentence B", "Sentence C"]

    def test_tts_stop_clears_pending_queue_items(self):
        tts = TextToSpeech()
        tts._engine = MagicMock()
        with tts._speech_lock:
            # Block the worker
            tts.speak_async("Item 1")
            tts.speak_async("Item 2")
            tts.speak_async("Item 3")
            tts.stop()
            # Queue should now be empty
            assert tts._speech_queue.empty()


# ============================================================================
# 4. Tool Handling During Streaming Tests
# ============================================================================

class DummyStreamingProvider(AIProvider):
    def __init__(self, stream_chunks, final_summary=""):
        self.stream_chunks = stream_chunks
        self.final_summary = final_summary
        self.generate_calls = 0
        self.stream_calls = 0

    def generate(self, messages):
        self.generate_calls += 1
        return self.final_summary

    def generate_stream(self, messages, cancel_event=None):
        self.stream_calls += 1
        if self.stream_calls == 1:
            for chunk in self.stream_chunks:
                if cancel_event and cancel_event.is_set():
                    break
                yield chunk
        else:
            if cancel_event and cancel_event.is_set():
                return
            yield self.final_summary


class FakeSafeTool(Tool):
    def __init__(self):
        super().__init__(
            name="fake_safe_tool",
            description="Safe test tool",
            parameters={"type": "object", "properties": {"message": {"type": "string"}}},
            permission_level=PermissionLevel.SAFE,
        )

    def execute(self, message: str = "ok") -> ToolResult:
        return ToolResult.ok(data={"echo": message})


class FakeDangerousTool(Tool):
    def __init__(self):
        super().__init__(
            name="fake_dangerous_tool",
            description="Dangerous test tool",
            parameters={"type": "object", "properties": {"action": {"type": "string"}}},
            permission_level=PermissionLevel.DANGEROUS,
        )

    def execute(self, action: str = "") -> ToolResult:
        return ToolResult.ok(data={"deleted": action})


class DenyConfirmationHandler(ConfirmationHandler):
    def request_confirmation(self, tool_name: str, description: str, permission_level: str) -> bool:
        return False


class TestStreamingToolSafety:
    def test_partial_tool_call_is_not_executed_and_never_spoken(self):
        registry = ToolRegistry()
        tool = FakeSafeTool()
        registry.register(tool)
        router = ToolRouter(registry, confirmation_handler=AutoApproveHandler())

        tool_json = '```tool_call\n{"tool": "fake_safe_tool", "arguments": {"message": "hello"}}\n```'
        # Stream in 3 separate chunks
        chunks = ['```tool_call\n{"tool": "fake_safe_', 'tool", "arguments": {"message": ', '"hello"}}\n```']
        provider = DummyStreamingProvider(chunks, final_summary="The tool executed successfully.")
        assistant = Assistant(provider, tool_router=router)

        yielded = list(assistant.send_stream("Run safe tool"))
        # The tool protocol should never be yielded to the caller/TTS
        assert all("tool_call" not in text for text in yielded)
        assert any("successfully" in text for text in yielded)

    def test_dangerous_tool_denied_confirmation_handled_safely(self):
        registry = ToolRegistry()
        registry.register(FakeDangerousTool())
        router = ToolRouter(registry, confirmation_handler=DenyConfirmationHandler())

        tool_json = '```tool_call\n{"tool": "fake_dangerous_tool", "arguments": {"action": "system_file"}}\n```'
        provider = DummyStreamingProvider([tool_json], final_summary="The action was declined by the user.")
        assistant = Assistant(provider, tool_router=router)

        yielded = list(assistant.send_stream("Delete file"))
        assert all("tool_call" not in text for text in yielded)
        assert any("declined permission" in text for text in yielded)


# ============================================================================
# 5. History Integrity Tests
# ============================================================================

class TestStreamingHistoryIntegrity:
    def test_final_response_committed_once_to_history(self):
        provider = DummyStreamingProvider(["Hello ", "world, ", "how are ", "you today?"])
        assistant = Assistant(provider)

        yielded = list(assistant.send_stream("Hello"))
        assert "".join(yielded) == "Hello world, how are you today?"

        # History should have: [system_prompt, assistant_ack, user_input, assistant_reply]
        history = assistant.history
        assert len(history) == 4
        assert history[-2]["role"] == "user"
        assert history[-2]["content"] == "Hello"
        assert history[-1]["role"] == "assistant"
        assert history[-1]["content"] == "Hello world, how are you today?"

    def test_streamed_tool_call_commits_complete_records_to_history(self):
        registry = ToolRegistry()
        registry.register(FakeSafeTool())
        router = ToolRouter(registry, confirmation_handler=AutoApproveHandler())

        tool_json = '```tool_call\n{"tool": "fake_safe_tool", "arguments": {"message": "ping"}}\n```'
        provider = DummyStreamingProvider([tool_json], final_summary="All done.")
        assistant = Assistant(provider, tool_router=router)

        list(assistant.send_stream("Do ping"))

        history = assistant.history
        # History sequence: [system, ack, user_req, assistant_tool_call, user_tool_result, assistant_final]
        assert len(history) == 6
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "Do ping"
        assert history[3]["role"] == "assistant"
        assert "```tool_call" in history[3]["content"]
        assert history[4]["role"] == "user"
        assert "```tool_result" in history[4]["content"]
        assert history[5]["role"] == "assistant"
        assert history[5]["content"] == "All done."


# ============================================================================
# 6. Voice Loop Streaming & Fallback Tests
# ============================================================================

class TestVoiceLoopStreaming:
    def test_voice_loop_streaming_enabled_speaks_sentences(self):
        assistant = MagicMock()
        assistant.send_stream.return_value = iter([
            "Good morning. ",
            "I have checked your system status. ",
            "Everything is operating normally.",
        ])
        stt = MagicMock()
        stt.device_name, stt.device_index = "Test mic", 0
        stt.listen.side_effect = ["check status", "exit"]
        tts = MagicMock()

        run_voice_mode(
            assistant,
            stt,
            tts,
            voice_streaming=True,
            stream_min_chars=10,
            stream_max_chars=100,
        )

        spoken = [call.args[0] for call in tts.speak.call_args_list]
        assert any("Good morning." in s for s in spoken)
        assert any("system status." in s for s in spoken)
        assert any("normally." in s for s in spoken)

    def test_voice_loop_streaming_failure_falls_back_to_non_streaming(self):
        assistant = MagicMock()
        # Streaming raises ProviderError
        assistant.send_stream.side_effect = ProviderError("SSE stream error")
        # Non-streaming generate succeeds
        assistant.send.return_value = "Fallback non-streaming response."

        stt = MagicMock()
        stt.device_name, stt.device_index = "Test mic", 0
        stt.listen.side_effect = ["hello", "exit"]
        tts = MagicMock()

        run_voice_mode(
            assistant,
            stt,
            tts,
            voice_streaming=True,
            stream_min_chars=10,
        )

        # assistant.send should have been called as fallback
        assistant.send.assert_called_once_with("hello")

    def test_voice_loop_streaming_disabled_uses_standard_pipeline(self):
        assistant = MagicMock()
        assistant.send.return_value = "Standard non-streaming reply."
        stt = MagicMock()
        stt.device_name, stt.device_index = "Test mic", 0
        stt.listen.side_effect = ["hello", "exit"]
        tts = MagicMock()

        run_voice_mode(
            assistant,
            stt,
            tts,
            voice_streaming=False,
        )

        assistant.send.assert_called_once_with("hello")
        assistant.send_stream.assert_not_called()

    def test_voice_loop_interruption_during_streaming_stops_speech(self):
        assistant = MagicMock()
        assistant.send_stream.return_value = iter(["Sentence one. ", "Sentence two."])
        tts = MagicMock()
        stt = MagicMock()

        # Call _speak_streamed_response with interruption
        with patch("main.is_tts_interrupt_command", return_value=True):
            tts.is_speaking = True
            stt.listen.return_value = "stop"
            # TextToSpeech mock instance for barge-in
            tts_real = TextToSpeech()
            tts_real._engine = MagicMock()
            interrupted = _speak_streamed_response(
                assistant=assistant,
                tts=tts_real,
                request="test",
                stt=stt,
                min_chars=5,
            )
            assert interrupted is True
