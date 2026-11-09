from unittest.mock import MagicMock

import pytest

from ai.provider import AIProvider, GeminiProvider, ProviderError
from config import Config
from core.assistant import Assistant
from core.confirmation import AutoApproveHandler
from core.tool_router import ToolRouter
from main import run_voice_mode
from tools.registry import ToolRegistry
from voice.stream_buffer import SentenceBuffer


def test_sentence_buffer_waits_for_natural_boundary():
    buffer = SentenceBuffer(min_chars=12, max_chars=40)
    assert buffer.feed("Hello, I can ") == []
    assert buffer.feed("help with that. Next") == ["Hello, I can help with that."]
    assert buffer.flush() == ["Next"]


def test_streaming_configuration_is_opt_in_and_bounded(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "fake")
    monkeypatch.setenv("AI_MODEL", "fake")
    monkeypatch.setenv("VOICE_STREAMING", "true")
    monkeypatch.setenv("VOICE_STREAM_MIN_CHARS", "50")
    monkeypatch.setenv("VOICE_STREAM_MAX_CHARS", "20")
    config = Config.load()
    assert config.voice_streaming is True
    assert config.voice_stream_min_chars == 50
    assert config.voice_stream_max_chars == 50


def test_sentence_buffer_forces_bounded_chunk_without_punctuation():
    buffer = SentenceBuffer(min_chars=5, max_chars=12)
    parts = buffer.feed("one two three four")
    assert parts[0] == "one two"
    assert "four" in " ".join(parts + buffer.flush())


def test_gemini_stream_parser_yields_text_and_ignores_bad_events():
    session = MagicMock()
    response = MagicMock()
    response.iter_lines.return_value = [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}',
        "data: not-json",
        'data: {"candidates":[{"content":{"parts":[{"text":"there."}]}}]}',
        "data: [DONE]",
    ]
    session.post.return_value = response
    provider = GeminiProvider("key", "model", session=session)
    assert list(provider.generate_stream([{"role": "user", "content": "hi"}])) == ["Hello ", "there."]
    assert session.post.call_args.kwargs["stream"] is True


def test_gemini_stream_timeout_becomes_provider_error():
    session = MagicMock()
    session.post.side_effect = __import__("requests").exceptions.Timeout()
    provider = GeminiProvider("key", "model", session=session)
    with pytest.raises(ProviderError, match="timed out"):
        list(provider.generate_stream([{"role": "user", "content": "hi"}]))


class StreamingProvider(AIProvider):
    def __init__(self, chunks, final=""):
        self.chunks = chunks
        self.final = final

    def generate(self, messages):
        return self.final

    def generate_stream(self, messages):
        yield from self.chunks


def test_assistant_stream_preserves_final_history():
    assistant = Assistant(StreamingProvider(["Hello ", "there."]))
    assert list(assistant.send_stream("hi")) == ["Hello ", "there."]
    assert assistant.history[-1]["content"] == "Hello there."


def test_streamed_tool_protocol_is_not_yielded_and_final_summary_is():
    provider = StreamingProvider(['```tool_call\n{"tool":"missing","arguments":{}}\n```'], final="The action could not be completed.")
    assistant = Assistant(
        provider,
        tool_router=ToolRouter(ToolRegistry(), confirmation_handler=AutoApproveHandler()),
    )
    chunks = list(assistant.send_stream("do something"))
    assert chunks == ["The requested tool is unavailable, so the action was not performed."]
    assert all("tool_call" not in chunk for chunk in chunks)


def test_voice_loop_streaming_queues_sentence_sized_text():
    assistant = MagicMock()
    assistant.send_stream.return_value = iter(["Hello, I can ", "help with that. Next step."])
    stt = MagicMock()
    stt.device_name, stt.device_index = "Test mic", 0
    stt.listen.side_effect = ["hello", "exit"]
    tts = MagicMock()
    run_voice_mode(assistant, stt, tts, voice_streaming=True, stream_min_chars=10)
    spoken = [call.args[0] for call in tts.speak.call_args_list]
    assert "Hello, I can help with that." in spoken
    assert "Next step." in spoken
