"""
Website Control Tools for AERIS.

Provides secure, allowlist-based opening of approved HTTPS websites in the default web browser.
Does not permit arbitrary URL navigation, unsafe URI schemes, or command execution.
"""

from __future__ import annotations

import logging
import re
import webbrowser
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry

logger = logging.getLogger("aeris.tools.websites")

# Explicit allowlist of approved websites and their canonical HTTPS URLs
WEBSITE_ALLOWLIST: Dict[str, Dict[str, Any]] = {
    "youtube": {
        "display_name": "YouTube",
        "url": "https://www.youtube.com/",
        "aliases": {"youtube", "yt", "youtube website", "youtube site"},
    },
    "google": {
        "display_name": "Google",
        "url": "https://www.google.com/",
        "aliases": {"google", "google search", "google website", "google site"},
    },
    "github": {
        "display_name": "GitHub",
        "url": "https://github.com/",
        "aliases": {"github", "git hub", "github website", "github site"},
    },
    "chatgpt": {
        "display_name": "ChatGPT",
        "url": "https://chatgpt.com/",
        "aliases": {"chatgpt", "chat gpt", "openai chatgpt", "chatgpt website", "chatgpt site"},
    },
    "gmail": {
        "display_name": "Gmail",
        "url": "https://mail.google.com/",
        "aliases": {"gmail", "google mail", "gmail website", "gmail site"},
    },
    "stackoverflow": {
        "display_name": "Stack Overflow",
        "url": "https://stackoverflow.com/",
        "aliases": {"stackoverflow", "stack overflow", "stack overflow website", "stackoverflow website"},
    },
    "google_drive": {
        "display_name": "Google Drive",
        "url": "https://drive.google.com/",
        "aliases": {"google drive", "gdrive", "google drive website", "google drive site"},
    },
    "microsoft_learn": {
        "display_name": "Microsoft Learn",
        "url": "https://learn.microsoft.com/",
        "aliases": {"microsoft learn", "ms learn", "microsoft learn website", "ms learn website"},
    },
    "wikipedia": {
        "display_name": "Wikipedia",
        "url": "https://www.wikipedia.org/",
        "aliases": {"wikipedia", "wiki", "wikipedia website", "wikipedia site"},
    },
}

_DANGEROUS_SCHEMES = re.compile(r"^(?:file|javascript|data|vbscript|about|chrome|brave|edge):", re.IGNORECASE)


class OpenWebsiteTool(Tool):
    """
    Tool to safely open approved websites from a predefined allowlist in the default browser.
    """
    name = "open_website"
    description = (
        "Safely opens an approved website from the allowlist in the default browser "
        "(e.g., 'youtube', 'google', 'github', 'chatgpt', 'gmail', 'stackoverflow', "
        "'google drive', 'microsoft learn', 'wikipedia')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "site_name": {
                "type": "string",
                "description": "Name or alias of the approved website to open (e.g. 'youtube', 'google', 'github').",
            }
        },
        "required": ["site_name"],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(
        self,
        allowlist: Optional[Dict[str, Dict[str, Any]]] = None,
        browser_opener: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__()
        self._allowlist = allowlist or WEBSITE_ALLOWLIST
        self._browser_opener = browser_opener
        self._alias_map: Dict[str, Dict[str, Any]] = {}
        self._build_alias_index()

    def _build_alias_index(self) -> None:
        """Construct a normalized lookup table from alias strings to site configurations."""
        self._alias_map.clear()
        for site_key, config in self._allowlist.items():
            entry = {"site_key": site_key, **config}
            self._alias_map[site_key.lower()] = entry
            display_name = config.get("display_name", "")
            if display_name:
                self._alias_map[display_name.lower().strip()] = entry
            for alias in config.get("aliases", []):
                self._alias_map[alias.lower().strip()] = entry

    def resolve_site(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a website identifier or alias to its allowlist configuration.
        Returns None if not in the approved allowlist.
        """
        if not user_query or not isinstance(user_query, str):
            return None

        normalized = user_query.strip().lower()
        if normalized in self._alias_map:
            return self._alias_map[normalized]

        # Try stripping trailing ' website' or ' site' qualifier
        stripped = re.sub(r"\s+(?:website|site)$", "", normalized).strip()
        if stripped in self._alias_map:
            return self._alias_map[stripped]

        return None

    def get_supported_aliases(self) -> List[str]:
        """Return a sorted list of all supported website names and aliases."""
        return sorted(list(self._alias_map.keys()))

    def execute(self, site_name: str = "", **kwargs: Any) -> ToolResult:
        """
        Open an approved allowlisted website in the default browser.

        Args:
            site_name: The website name or alias to open.
        """
        if not site_name or not isinstance(site_name, str) or not site_name.strip():
            return ToolResult.fail(
                error="INVALID_ARGUMENTS",
                message="Website name must be a non-empty string.",
            )

        trimmed = site_name.strip()

        # Reject dangerous schemes and raw arbitrary URLs immediately
        if _DANGEROUS_SCHEMES.search(trimmed):
            logger.warning("Rejected attempt to open unsafe URI scheme: '%s'", trimmed)
            return ToolResult.fail(
                error="UNSUPPORTED_SCHEME",
                message="Dangerous or non-HTTP schemes are not permitted.",
            )

        # Disallow raw URLs that are not explicitly matched to an allowlist canonical entry
        if trimmed.startswith("http://") or trimmed.startswith("https://"):
            parsed = urlparse(trimmed)
            # Find if this raw URL matches an exact approved canonical URL
            matched_config = None
            for config in self._allowlist.values():
                if config.get("url", "").rstrip("/") == trimmed.rstrip("/"):
                    matched_config = config
                    break
            if not matched_config:
                logger.warning("Rejected attempt to open arbitrary user-supplied URL: '%s'", trimmed)
                return ToolResult.fail(
                    error="UNSUPPORTED_WEBSITE",
                    message="Arbitrary user-supplied URLs are not permitted. Only approved allowlisted websites can be opened.",
                )
            site_info = matched_config
        else:
            site_info = self.resolve_site(trimmed)

        if not site_info:
            logger.warning("Rejected attempt to open non-allowlisted website: '%s'", trimmed)
            return ToolResult.fail(
                error="UNKNOWN_WEBSITE",
                message=f"Website '{trimmed}' is not in the approved allowlist.",
                data={"requested": trimmed, "supported": sorted(list(self._allowlist.keys()))},
            )

        url = site_info.get("url", "")
        # Extra safeguard: ensure URL is HTTPS and non-empty
        if not url.startswith("https://"):
            return ToolResult.fail(
                error="INSECURE_URL",
                message=f"Approved URL for '{site_info.get('display_name')}' must use HTTPS.",
            )

        display_name = site_info.get("display_name", site_info.get("site_key", "website"))

        try:
            logger.info("Opening approved website '%s' at URL: '%s'", display_name, url)
            opener = self._browser_opener if self._browser_opener is not None else webbrowser.open
            opener_result = opener(url)

            # Some custom openers return False on failure
            if opener_result is False:
                return ToolResult.fail(
                    error="LAUNCH_FAILED",
                    message=f"Failed to open '{display_name}' in the default browser.",
                )

            return ToolResult.ok(
                message=f"Opened {display_name} ({url}).",
                data={
                    "site_key": site_info.get("site_key"),
                    "display_name": display_name,
                    "url": url,
                },
            )
        except Exception as exc:
            logger.exception("Unexpected error opening website '%s': %s", display_name, exc)
            return ToolResult.fail(
                error="LAUNCH_FAILED",
                message=f"Failed to open '{display_name}': {exc}",
            )


def register_website_tools(registry: ToolRegistry) -> None:
    """
    Convenience function to register OpenWebsiteTool with a ToolRegistry.
    """
    registry.register(OpenWebsiteTool(), overwrite=True)
