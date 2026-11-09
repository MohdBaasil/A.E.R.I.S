"""Conservative deterministic routing for a small set of clear user intents."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from core.tool_router import ToolCall


@dataclass(frozen=True)
class IntentStep:
    """One proposed registered-tool action and its prior-step dependencies."""

    intent: str
    call: ToolCall
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True)
class IntentRoute:
    """High-confidence intent classification; this object never executes tools."""

    intent: str
    confidence: float
    reason: str
    steps: tuple[IntentStep, ...]


_EDUCATIONAL = re.compile(
    r"\b(?:how\s+does|explain|define|what\s+does|"
    r"what\s+do\s+you\s+mean\s+by)\b",
    re.IGNORECASE,
)
_SHELL_OR_COMMAND = re.compile(
    r"\b(?:powershell|pwsh|cmd(?:\.exe)?|command\s+prompt|terminal|bash|zsh|shell)\s+\S+"
    r"|^\s*(?:powershell|pwsh|cmd(?:\.exe)?|command\s+prompt|terminal|bash|zsh|shell)\s*$"
    r"|\b(?:execute|run\s+(?:this\s+)?command|run\s+(?:powershell|pwsh|cmd|bash|zsh))\b"
    r"|[|`;$\n\r]",
    re.IGNORECASE,
)
_APP_VERB = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:open|launch|start)\s+(?:the\s+)?(.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_APP_NON_TARGET = re.compile(r"\b(?:file|files|folder|directory|project|website|url|link)\b", re.IGNORECASE)
_WEBSITE_VERB = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:open|launch|start|go\s+to|visit)\s+(?:the\s+)?(.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_KNOWN_WEBSITES = frozenset({
    "youtube", "yt", "youtube website", "youtube site",
    "google", "google search", "google website", "google site",
    "github", "git hub", "github website", "github site",
    "chatgpt", "chat gpt", "openai chatgpt", "chatgpt website", "chatgpt site",
    "gmail", "google mail", "gmail website", "gmail site",
    "stackoverflow", "stack overflow", "stackoverflow website", "stack overflow website",
    "google drive", "gdrive", "google drive website", "google drive site",
    "microsoft learn", "ms learn", "microsoft learn website", "ms learn website",
    "wikipedia", "wiki", "wikipedia website", "wikipedia site",
})
_FILE_VERB = re.compile(r"\b(?:list|show|display)\b|\bwhat\s+files\s+are\b", re.IGNORECASE)
_PATH_SHAPE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|\.{1,2}[\\/]|~[\\/])")

_AUDIO_VOLUME_UP = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:turn\s+up\s+(?:the\s+)?volume|volume\s+up|increase\s+(?:the\s+)?volume|raise\s+(?:the\s+)?volume|make\s+it\s+louder|louder)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_AUDIO_VOLUME_DOWN = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:turn\s+down\s+(?:the\s+)?volume|volume\s+down|decrease\s+(?:the\s+)?volume|lower\s+(?:the\s+)?volume|make\s+it\s+(?:quieter|softer)|quieter|softer)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_AUDIO_UNMUTE = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:unmute(?:\s+(?:the\s+)?(?:audio|sound|volume|system))?|turn\s+on\s+(?:the\s+)?sound)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_AUDIO_MUTE = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:mute(?:\s+(?:the\s+)?(?:audio|sound|volume|system))?|turn\s+off\s+(?:the\s+)?sound)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_AUDIO_VERB = re.compile(
    r"\b(?:mute|unmute|volume\s+up|volume\s+down|louder|quieter|softer)\b",
    re.IGNORECASE,
)

_MEDIA_PLAY_PAUSE = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:"
    r"(?:play\s+or\s+pause|toggle)(?:\s+(?:my\s+|the\s+)?(?:media\s+playback|media|playback|music|song|track))?"
    r"|pause(?:\s+(?:the\s+|my\s+)?(?:media\s+playback|music|playback|media|song|track))?"
    r"|resume(?:\s+(?:the\s+|my\s+)?(?:media\s+playback|music|playback|media|song|track))?"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_MEDIA_NEXT_TRACK = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:"
    r"(?:skip|play\s+the|play|next)\s+(?:this\s+|the\s+|to\s+the\s+)?(?:next\s+)?(?:track|song)"
    r"|next\s+(?:track|song)"
    r"|skip\s+(?:ahead|this\s+one|forward)"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_MEDIA_PREV_TRACK = re.compile(
    r"^\s*(?:could\s+you\s+(?:please\s+)?|please\s+)?(?:"
    r"(?:play\s+the|play|previous|prev)\s+(?:the\s+|to\s+the\s+)?(?:previous|prev|last)\s+(?:track|song)"
    r"|previous\s+(?:track|song)"
    r"|prev\s+(?:track|song)"
    r"|last\s+(?:track|song)"
    r"|go\s+back\s+(?:one|a)\s+(?:track|song)"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_MEDIA_VERB = re.compile(
    r"\b(?:play\s+or\s+pause|toggle(?:\s+media|\s+playback)?|pause|resume|next\s+track|next\s+song|skip\s+track|skip\s+song|previous\s+track|previous\s+song)\b",
    re.IGNORECASE,
)


class IntentRouter:
    """Map only unambiguous system, audio, media, file-list, and app-launch requests to tool calls."""

    MAX_STEPS = 4

    def route(self, text: str) -> Optional[IntentRoute]:
        if not isinstance(text, str) or not text.strip():
            return None
        if _SHELL_OR_COMMAND.search(text):
            return None

        # A system-information question is treated as one snapshot even when it
        # mentions several metrics, rather than being split into parallel calls.
        has_other_action = bool(
            _APP_VERB.search(text) or _FILE_VERB.search(text) or _AUDIO_VERB.search(text) or _MEDIA_VERB.search(text) or _WEBSITE_VERB.search(text)
        )
        system_call = None if has_other_action else self._system_info_call(text)
        if system_call is not None:
            return IntentRoute(
                intent="SYSTEM_INFO", confidence=0.99,
                reason="current system metric or status request", 
                steps=(IntentStep("SYSTEM_INFO", system_call),),
            )

        clauses = self._split_ordered_actions(text)
        if len(clauses) > 1:
            if len(clauses) > self.MAX_STEPS:
                return None
            steps: list[IntentStep] = []
            for index, clause in enumerate(clauses):
                step = self._single_step(clause, allow_contextual_directory=True)
                if step is None:
                    return None
                dependencies = (index - 1,) if index else ()
                steps.append(IntentStep(step.intent, step.call, dependencies))
            return IntentRoute(
                intent="MULTI_ACTION", confidence=0.95,
                reason="explicit ordered actions", steps=tuple(steps),
            )

        step = self._single_step(text, allow_contextual_directory=False)
        if step is None:
            return None
        return IntentRoute(
            intent=step.intent, confidence=0.99,
            reason=f"high-confidence {step.intent.lower()} request", steps=(step,),
        )

    @staticmethod
    def _split_ordered_actions(text: str) -> list[str]:
        # Split only explicit sequencing/conjunctions. Metric conjunctions have
        # already been recognized as one SYSTEM_INFO intent above.
        parts = re.split(r"\s+(?:and\s+then|then|and)\s+", text.strip(), flags=re.IGNORECASE)
        return [part.strip(" ,") for part in parts if part.strip(" ,")]

    def _single_step(self, text: str, *, allow_contextual_directory: bool) -> Optional[IntentStep]:
        call = self._system_info_call(text)
        if call is not None:
            return IntentStep("SYSTEM_INFO", call)

        audio_call = self._system_audio_call(text)
        if audio_call is not None:
            return IntentStep("SYSTEM_AUDIO", audio_call)

        media_call = self._system_media_call(text)
        if media_call is not None:
            return IntentStep("SYSTEM_MEDIA", media_call)

        file_call = self._list_files_call(text, allow_contextual_directory=allow_contextual_directory)
        if file_call is not None:
            return IntentStep("LIST_FILES", file_call)

        website_call = self._open_website_call(text)
        if website_call is not None:
            return IntentStep("OPEN_WEBSITE", website_call)

        app_call = self._open_application_call(text)
        if app_call is not None:
            return IntentStep("OPEN_APPLICATION", app_call)
        return None

    @staticmethod
    def _system_info_call(text: str) -> Optional[ToolCall]:
        query = re.sub(r"\s+", " ", text.casefold()).strip()
        if not query or _EDUCATIONAL.search(query):
            return None
        if re.search(r"\b(?:technology|definition|meaning|health|what it is)\b", query):
            return None
        if re.search(r"\b(?:call|use|invoke)\s+(?:the\s+)?(?:registered\s+)?system_info(?:\s+tool)?\b", query):
            # Preserve the model-originated/native tool-call path when the user
            # explicitly asks to exercise that protocol.
            return None
        # Pure conceptual questions like "What is CPU?" lack metric/state indicators
        # (usage, load, status, etc.) and belong to the LLM.
        if re.match(r"^what\s+is\b", query) and not re.search(
            r"\b(?:my|current|currently|this|usage|utili[sz]ation|load|status|level|charge|state|percent|remaining|free)\b",
            query,
        ):
            return None

        system_topic = bool(re.search(
            r"\b(?:system information|system info|computer information|computer specs|system specs|my system)\b",
            query,
        ))
        metric_topic = bool(re.search(r"\b(?:cpu|processor|ram|memory|disk|storage|drive|battery|charge|power)\b", query))
        current_action = bool(re.search(
            r"\b(?:check|show|get|display|report|current|currently|usage|utili[sz]ation|load|using|used|"
            r"remaining|level|charge|status|life|percent(?:age)?|state)\b|\bhow much\b|\bright now\b|\bwhat(?:'s|\s+is)?(?:\s+my)?\b",
            query,
        )) or bool(re.fullmatch(r"tell me about (?:my )?battery", query))

        if system_topic and current_action:
            return ToolCall(tool="system_info", arguments={})
        if metric_topic and current_action:
            return ToolCall(tool="system_info", arguments={})
        return None

    @staticmethod
    def _list_files_call(text: str, *, allow_contextual_directory: bool) -> Optional[ToolCall]:
        query = re.sub(r"\s+", " ", text.strip()).strip(" .!?\t")
        if not _FILE_VERB.search(query):
            return None

        current_directory = re.search(
            r"\b(?:here|in here|this folder|this directory|current folder|current directory|"
            r"working directory|my project files|files in my project|files in the project)\b",
            query,
            re.IGNORECASE,
        )
        if current_directory or (allow_contextual_directory and re.fullmatch(
            r"(?:list|show|display)\s+(?:the\s+)?files|what\s+files\s+are\s+there",
            query,
            re.IGNORECASE,
        )):
            return ToolCall(tool="list_files", arguments={"path": os.getcwd()})

        # Accept a directory only when the user explicitly supplied a quoted
        # path or a recognizable absolute/relative path. Do not infer one.
        path_match = re.search(
            r"\b(?:in|from|under)\s+(?P<path>'[^']+'|\"[^\"]+\"|[^?]+?)\s*$",
            query,
            re.IGNORECASE,
        )
        if path_match:
            raw_path = path_match.group("path").strip().rstrip(".,!? ")
            if len(raw_path) >= 2 and raw_path[0] in "'\"" and raw_path[-1] == raw_path[0]:
                raw_path = raw_path[1:-1]
                return ToolCall(tool="list_files", arguments={"path": raw_path}) if raw_path else None
            if _PATH_SHAPE.match(raw_path):
                return ToolCall(tool="list_files", arguments={"path": raw_path})
            # Phrases such as "in this directory" are explicit aliases handled
            # above; a bare project/folder name is not enough to guess a path.
        return None

    @staticmethod
    def _open_application_call(text: str) -> Optional[ToolCall]:
        if _SHELL_OR_COMMAND.search(text):
            return None
        match = _APP_VERB.fullmatch(text)
        if not match:
            return None
        app_name = match.group(1).strip()
        if not app_name or _APP_NON_TARGET.search(app_name) or any(char in app_name for char in "\\/|&`$;\n\r"):
            return None
        return ToolCall(tool="open_application", arguments={"app_name": app_name})

    @staticmethod
    def _system_audio_call(text: str) -> Optional[ToolCall]:
        query = text.strip()
        if not query or _EDUCATIONAL.search(query) or _SHELL_OR_COMMAND.search(query):
            return None
        if _AUDIO_VOLUME_UP.match(query):
            return ToolCall(tool="system_audio", arguments={"action": "volume_up"})
        if _AUDIO_VOLUME_DOWN.match(query):
            return ToolCall(tool="system_audio", arguments={"action": "volume_down"})
        if _AUDIO_UNMUTE.match(query):
            return ToolCall(tool="system_audio", arguments={"action": "unmute"})
        if _AUDIO_MUTE.match(query):
            return ToolCall(tool="system_audio", arguments={"action": "mute"})
        return None

    @staticmethod
    def _system_media_call(text: str) -> Optional[ToolCall]:
        query = text.strip()
        if not query or _EDUCATIONAL.search(query) or _SHELL_OR_COMMAND.search(query):
            return None
        # Reject requests explicitly targeting an application, game, or video
        if re.search(r"\b(?:in|on|using|with)\s+[a-zA-Z0-9_-]+|\bgame\b|\bvideo\b", query, re.IGNORECASE):
            return None
        if _MEDIA_PLAY_PAUSE.match(query):
            return ToolCall(tool="system_media", arguments={"action": "play_pause"})
        if _MEDIA_NEXT_TRACK.match(query):
            return ToolCall(tool="system_media", arguments={"action": "next_track"})
        if _MEDIA_PREV_TRACK.match(query):
            return ToolCall(tool="system_media", arguments={"action": "previous_track"})
        return None

    @staticmethod
    def _open_website_call(text: str) -> Optional[ToolCall]:
        if not text or not isinstance(text, str):
            return None
        if _EDUCATIONAL.search(text) or _SHELL_OR_COMMAND.search(text):
            return None
        match = _WEBSITE_VERB.fullmatch(text)
        if not match:
            return None
        target = match.group(1).strip()
        if not target or any(char in target for char in "|&`$;\n\r"):
            return None

        normalized = target.lower()
        clean_target = re.sub(r"\s+(?:website|site)$", "", normalized).strip()

        # Check if matched against known allowlisted websites
        if normalized in _KNOWN_WEBSITES or clean_target in _KNOWN_WEBSITES:
            return ToolCall(tool="open_website", arguments={"site_name": target})

        # Explicit website keywords (e.g., "open random_site website" or raw URL "open https://...")
        is_explicit_website = bool(
            re.search(r"\b(?:website|site)\b", target, re.IGNORECASE)
            or target.startswith(("http://", "https://", "file:", "javascript:", "data:"))
        )
        if is_explicit_website:
            return ToolCall(tool="open_website", arguments={"site_name": target})

        return None
