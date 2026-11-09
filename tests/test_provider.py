import pytest
import requests
from unittest.mock import patch, MagicMock

from ai.provider import GeminiProvider, ProviderError, create_provider
from config import Config


def test_create_provider_returns_gemini_for_known_provider():
    config = Config(ai_provider="gemini", ai_api_key="fake-key", ai_model="gemini-3.5-flash-lite")
    provider = create_provider(config)
    assert isinstance(provider, GeminiProvider)


def test_create_provider_rejects_unknown_provider():
    config = Config(ai_provider="not-a-real-provider", ai_api_key="fake-key", ai_model="x")
    with pytest.raises(ProviderError):
        create_provider(config)


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_success(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hello there!"}]}}]
    }
    mock_post.return_value = mock_response

    result = provider.generate([{"role": "user", "content": "hi"}])
    assert result == "Hello there!"


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_503_retry_success(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    
    # First response fails with 503
    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=503))
    
    # Second response succeeds
    success_response = MagicMock()
    success_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Hello there after retry!"}]}}]
    }
    
    mock_post.side_effect = [fail_response, success_response]

    result = provider.generate([{"role": "user", "content": "hi"}])
    assert result == "Hello there after retry!"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_503_repeated_failure(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    
    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=503))
    
    mock_post.return_value = fail_response
    mock_post.side_effect = None

    with pytest.raises(ProviderError, match="temporarily unavailable"):
        provider.generate([{"role": "user", "content": "hi"}])
    
    assert mock_post.call_count == 3


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_401_no_retry(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    
    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=401))
    mock_post.return_value = fail_response

    with pytest.raises(ProviderError, match="Authentication failed"):
        provider.generate([{"role": "user", "content": "hi"}])
    
    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_404(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    
    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=404))
    mock_post.return_value = fail_response

    with pytest.raises(ProviderError, match="not found"):
        provider.generate([{"role": "user", "content": "hi"}])


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_429(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    
    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=429))
    mock_post.return_value = fail_response

    with pytest.raises(ProviderError, match="Rate limit exceeded"):
        provider.generate([{"role": "user", "content": "hi"}])


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_timeout_retry(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    
    mock_post.side_effect = requests.exceptions.Timeout("Timeout")

    with pytest.raises(ProviderError, match="timed out"):
        provider.generate([{"role": "user", "content": "hi"}])
    
    assert mock_post.call_count == 3


@patch("ai.provider.time.sleep", return_value=None)
@patch("ai.provider.requests.Session.post")
def test_generate_malformed_response(mock_post, mock_sleep):
    provider = GeminiProvider(api_key="fake-key", model="gemini-3.5-flash-lite")
    mock_response = MagicMock()
    mock_response.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_response

    with pytest.raises(ProviderError, match="unexpected response"):
        provider.generate([{"role": "user", "content": "hi"}])
