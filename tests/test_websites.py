"""
Automated unit tests for OpenWebsiteTool and website intent routing (Phase 10D).

All browser-opening operations are mocked; no real browser windows or network
requests are initiated during test execution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.intent_router import IntentRouter
from core.tool_router import ToolCall, ToolRouter
from core.tool_setup import create_tool_registry
from tools.base import ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry
from tools.websites import OpenWebsiteTool, WEBSITE_ALLOWLIST, register_website_tools


# ==============================================================================
# 1. Canonical Allowlist & Alias Resolution Tests
# ==============================================================================

EXPECTED_CANONICAL_SITES = [
    ("youtube", "https://www.youtube.com/", "YouTube"),
    ("google", "https://www.google.com/", "Google"),
    ("github", "https://github.com/", "GitHub"),
    ("chatgpt", "https://chatgpt.com/", "ChatGPT"),
    ("gmail", "https://mail.google.com/", "Gmail"),
    ("stackoverflow", "https://stackoverflow.com/", "Stack Overflow"),
    ("google drive", "https://drive.google.com/", "Google Drive"),
    ("microsoft learn", "https://learn.microsoft.com/", "Microsoft Learn"),
    ("wikipedia", "https://www.wikipedia.org/", "Wikipedia"),
]


@pytest.mark.parametrize("query,expected_url,expected_name", EXPECTED_CANONICAL_SITES)
def test_all_canonical_websites_resolve_correctly(query, expected_url, expected_name):
    tool = OpenWebsiteTool()
    resolved = tool.resolve_site(query)

    assert resolved is not None
    assert resolved["url"] == expected_url
    assert resolved["display_name"] == expected_name


@pytest.mark.parametrize(
    "alias,expected_key",
    [
        ("youtube", "youtube"),
        ("YouTube", "youtube"),
        ("YT", "youtube"),
        ("youtube website", "youtube"),
        ("google", "google"),
        ("Google", "google"),
        ("google search", "google"),
        ("google website", "google"),
        ("github", "github"),
        ("GitHub", "github"),
        ("git hub", "github"),
        ("github website", "github"),
        ("chatgpt", "chatgpt"),
        ("ChatGPT", "chatgpt"),
        ("chat gpt", "chatgpt"),
        ("openai chatgpt", "chatgpt"),
        ("chatgpt website", "chatgpt"),
        ("gmail", "gmail"),
        ("Gmail", "gmail"),
        ("google mail", "gmail"),
        ("gmail website", "gmail"),
        ("stackoverflow", "stackoverflow"),
        ("Stack Overflow", "stackoverflow"),
        ("stack overflow", "stackoverflow"),
        ("stackoverflow website", "stackoverflow"),
        ("google drive", "google_drive"),
        ("Google Drive", "google_drive"),
        ("gdrive", "google_drive"),
        ("google drive website", "google_drive"),
        ("microsoft learn", "microsoft_learn"),
        ("Microsoft Learn", "microsoft_learn"),
        ("ms learn", "microsoft_learn"),
        ("microsoft learn website", "microsoft_learn"),
        ("wikipedia", "wikipedia"),
        ("Wikipedia", "wikipedia"),
        ("wiki", "wikipedia"),
        ("wikipedia website", "wikipedia"),
    ],
)
def test_website_aliases_resolve_to_correct_keys(alias, expected_key):
    tool = OpenWebsiteTool()
    resolved = tool.resolve_site(alias)

    assert resolved is not None
    assert resolved["site_key"] == expected_key


# ==============================================================================
# 2. Execution & Dispatch Tests (Mocked Browser Opener)
# ==============================================================================

@pytest.mark.parametrize("query,expected_url,expected_name", EXPECTED_CANONICAL_SITES)
def test_open_website_dispatches_canonical_url(query, expected_url, expected_name):
    mock_opener = MagicMock(return_value=True)
    tool = OpenWebsiteTool(browser_opener=mock_opener)

    result = tool.run(site_name=query)

    assert result.success is True
    assert result.error is None
    assert expected_url in result.message
    assert result.data["url"] == expected_url
    assert result.data["display_name"] == expected_name
    mock_opener.assert_called_once_with(expected_url)


def test_open_website_handles_opener_failure_cleanly():
    def failing_opener(url):
        raise OSError("Browser launch failed")

    tool = OpenWebsiteTool(browser_opener=failing_opener)
    result = tool.run(site_name="github")

    assert result.success is False
    assert result.error == "LAUNCH_FAILED"
    assert "Failed to open 'GitHub'" in result.message


def test_open_website_handles_opener_returning_false():
    mock_opener = MagicMock(return_value=False)
    tool = OpenWebsiteTool(browser_opener=mock_opener)

    result = tool.run(site_name="github")

    assert result.success is False
    assert result.error == "LAUNCH_FAILED"


# ==============================================================================
# 3. Security, Dangerous Schemes, & Arbitrary URL Rejections
# ==============================================================================

@pytest.mark.parametrize(
    "unsafe_input",
    [
        "file:///C:/Windows/System32/calc.exe",
        "file://etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>evil()</script>",
        "vbscript:msgbox",
        "about:blank",
        "chrome://settings",
    ],
)
def test_open_website_rejects_dangerous_schemes(unsafe_input):
    mock_opener = MagicMock()
    tool = OpenWebsiteTool(browser_opener=mock_opener)

    result = tool.run(site_name=unsafe_input)

    assert result.success is False
    assert result.error == "UNSUPPORTED_SCHEME"
    mock_opener.assert_not_called()


@pytest.mark.parametrize(
    "arbitrary_url",
    [
        "https://unknown-example.invalid",
        "https://evil-phishing-site.com",
        "http://insecure-site.org",
        "https://malicious.example.com/payload.exe",
    ],
)
def test_open_website_rejects_arbitrary_urls(arbitrary_url):
    mock_opener = MagicMock()
    tool = OpenWebsiteTool(browser_opener=mock_opener)

    result = tool.run(site_name=arbitrary_url)

    assert result.success is False
    assert result.error == "UNSUPPORTED_WEBSITE"
    mock_opener.assert_not_called()


def test_open_website_rejects_empty_name():
    mock_opener = MagicMock()
    tool = OpenWebsiteTool(browser_opener=mock_opener)

    result = tool.run(site_name="   ")

    assert result.success is False
    assert result.error == "INVALID_ARGUMENTS"
    mock_opener.assert_not_called()


def test_open_website_rejects_unknown_website():
    mock_opener = MagicMock()
    tool = OpenWebsiteTool(browser_opener=mock_opener)

    result = tool.run(site_name="some_unsupported_site")

    assert result.success is False
    assert result.error == "UNKNOWN_WEBSITE"
    assert "not in the approved allowlist" in result.message
    mock_opener.assert_not_called()


# ==============================================================================
# 4. Metadata, Schema, & Registration Tests
# ==============================================================================

def test_open_website_metadata_and_schema():
    tool = OpenWebsiteTool()
    assert tool.name == "open_website"
    assert tool.permission_level == PermissionLevel.SAFE
    assert "site_name" in tool.parameters["properties"]

    schema = tool.get_schema()
    assert schema["name"] == "open_website"
    assert schema["permission_level"] == "SAFE"
    assert "site_name" in schema["parameters"]["properties"]


def test_open_website_registers_in_catalog():
    registry = create_tool_registry()

    assert registry.has("open_website") is True
    tool = registry.get("open_website")
    assert isinstance(tool, OpenWebsiteTool)
    assert "open_website" in registry.list_names()
    # Confirm existing tools remain registered
    assert "open_application" in registry.list_names()
    assert "system_audio" in registry.list_names()
    assert "system_media" in registry.list_names()


def test_register_website_tools_is_idempotent():
    registry = ToolRegistry()
    register_website_tools(registry)
    register_website_tools(registry)

    assert registry.has("open_website")
    assert len([name for name in registry.list_names() if name == "open_website"]) == 1


# ==============================================================================
# 5. Deterministic Intent Routing Tests
# ==============================================================================

@pytest.mark.parametrize(
    "phrase,expected_site_arg",
    [
        ("open YouTube", "YouTube"),
        ("open Google", "Google"),
        ("open GitHub", "GitHub"),
        ("open ChatGPT", "ChatGPT"),
        ("open Gmail", "Gmail"),
        ("open Stack Overflow", "Stack Overflow"),
        ("open Google Drive", "Google Drive"),
        ("open Microsoft Learn", "Microsoft Learn"),
        ("open Wikipedia", "Wikipedia"),
        ("open YouTube website", "YouTube website"),
        ("launch GitHub website", "GitHub website"),
        ("open Wikipedia website", "Wikipedia website"),
        ("open Google Drive website", "Google Drive website"),
        ("go to YouTube", "YouTube"),
        ("visit GitHub", "GitHub"),
    ],
)
def test_intent_router_recognizes_clear_website_requests(phrase, expected_site_arg):
    router = IntentRouter()
    route = router.route(phrase)

    assert route is not None
    assert route.intent == "OPEN_WEBSITE"
    assert len(route.steps) == 1
    assert route.steps[0].intent == "OPEN_WEBSITE"
    assert route.steps[0].call.tool == "open_website"
    assert route.steps[0].call.arguments == {"site_name": expected_site_arg}


@pytest.mark.parametrize(
    "phrase",
    [
        "What is YouTube?",
        "How do websites work?",
        "How does Google search work?",
        "Explain Wikipedia.",
        "What is ChatGPT?",
        "How does ChatGPT work?",
        "Define GitHub.",
    ],
)
def test_intent_router_does_not_intercept_educational_website_questions(phrase):
    router = IntentRouter()
    route = router.route(phrase)

    if route is not None:
        assert route.intent != "OPEN_WEBSITE"


# ==============================================================================
# 6. Disambiguation with Application Launching & Other Tools
# ==============================================================================

def test_intent_router_distinguishes_applications_from_websites():
    router = IntentRouter()

    # Notepad -> OPEN_APPLICATION
    route_np = router.route("open notepad")
    assert route_np is not None
    assert route_np.intent == "OPEN_APPLICATION"
    assert route_np.steps[0].call.arguments == {"app_name": "notepad"}

    # Brave Browser -> OPEN_APPLICATION
    route_brave = router.route("open brave browser")
    assert route_brave is not None
    assert route_brave.intent == "OPEN_APPLICATION"
    assert route_brave.steps[0].call.arguments == {"app_name": "brave browser"}

    # Google Chrome -> OPEN_APPLICATION
    route_chrome = router.route("open Google Chrome")
    assert route_chrome is not None
    assert route_chrome.intent == "OPEN_APPLICATION"
    assert route_chrome.steps[0].call.arguments == {"app_name": "Google Chrome"}

    # Google (alone) -> OPEN_WEBSITE
    route_google = router.route("open Google")
    assert route_google is not None
    assert route_google.intent == "OPEN_WEBSITE"
    assert route_google.steps[0].call.arguments == {"site_name": "Google"}


def test_play_a_game_and_media_controls_remain_unaffected():
    router = IntentRouter()

    # "play a game" must not be routed to media or website
    route_game = router.route("play a game")
    if route_game is not None:
        assert route_game.intent not in ("SYSTEM_MEDIA", "OPEN_WEBSITE")

    # media commands must still route to SYSTEM_MEDIA
    route_media = router.route("pause playback")
    assert route_media is not None
    assert route_media.intent == "SYSTEM_MEDIA"


# ==============================================================================
# 7. End-to-End Assistant Integration Tests (Text and Stream)
# ==============================================================================

def test_assistant_executes_website_opening_deterministically():
    from core.assistant import Assistant
    from ai.provider import AIProvider

    class MockProvider(AIProvider):
        def generate(self, messages):
            return "fallback"

    mock_opener = MagicMock(return_value=True)
    registry = ToolRegistry()
    registry.register(OpenWebsiteTool(browser_opener=mock_opener))
    router = ToolRouter(registry)

    assistant = Assistant(provider=MockProvider(), tool_router=router)

    reply = assistant.send("open YouTube")
    assert "Opened YouTube (https://www.youtube.com/)." in reply
    mock_opener.assert_called_once_with("https://www.youtube.com/")


def test_assistant_streaming_executes_website_opening():
    from core.assistant import Assistant
    from ai.provider import AIProvider

    class MockProvider(AIProvider):
        def generate(self, messages):
            return "fallback"
        def generate_stream(self, messages, cancel_event=None):
            yield "fallback"

    mock_opener = MagicMock(return_value=True)
    registry = ToolRegistry()
    registry.register(OpenWebsiteTool(browser_opener=mock_opener))
    router = ToolRouter(registry)

    assistant = Assistant(provider=MockProvider(), tool_router=router)

    chunks = list(assistant.send_stream("open GitHub"))
    full_reply = "".join(chunks)
    assert "Opened GitHub (https://github.com/)." in full_reply
    mock_opener.assert_called_once_with("https://github.com/")


def test_assistant_rejects_unsupported_website_cleanly():
    from core.assistant import Assistant
    from ai.provider import AIProvider

    class MockProvider(AIProvider):
        def generate(self, messages):
            return "fallback"

    mock_opener = MagicMock()
    registry = ToolRegistry()
    registry.register(OpenWebsiteTool(browser_opener=mock_opener))
    router = ToolRouter(registry)

    assistant = Assistant(provider=MockProvider(), tool_router=router)

    reply = assistant.send("open random_website website")
    assert "That website is not in the approved allowlist, so it was not opened." in reply
    mock_opener.assert_not_called()
