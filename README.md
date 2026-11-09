# A.E.R.I.S.

**Adaptive Electronic Reasoning & Intelligence System**

A personal AI desktop assistant built specifically for Windows. Developed incrementally, phase by phase.

## Current Status: Phase 10 — Windows Desktop Control & System Automation

A.E.R.I.S. is an intelligent, multimodal desktop assistant featuring:
- **True Voice-to-Voice Loop**: Speak → AERIS listens → AI generates → AERIS speaks back → listens again. Fully hands-free.
- **Real-Time Streaming Voice**: Incremental Server-Sent Events (SSE) token streaming coupled with natural sentence buffering for low-latency speech synthesis.
- **Local Intent Interception**: Media playback, master volume/mute, app launches, and safe website visits are executed locally without cloud round-trips.
- **Persistent Bounded Memory**: SQLite-backed semantic and episodic memory with automated credential and PII sanitization.
- **Permission Safety Gates**: Categorization of actions into `SAFE`, `CONFIRM`, and `DANGEROUS` with interactive confirmation via terminal prompt or spoken voice.
- **Extensible Desktop Tool System**: 12 registered tools covering system diagnostics, audio control, media playback, application launching, allowlisted website navigation, and safe file management.
- **Dual AI Provider Architecture**: Seamless support for Google Gemini (cloud REST/SSE) and Ollama (local offline LLMs).
- **Graceful Hardware Degradation**: Automatic microphone device discovery, noise calibration, and fallback to terminal text mode if audio input is unavailable.

---

## Architecture Overview

```
AERIS/
├── main.py                 # Entry point: text/voice loops, CLI flags, local command interception
├── config.py               # Centralized configuration with validation & defaults
├── requirements.txt        # Pinned project dependencies
├── .env.example            # Configuration template with placeholder values
├── aeris.log               # Operational runtime log (no secrets logged)
│
├── core/
│   ├── assistant.py        # Assistant orchestration, conversation memory, history pruning
│   ├── confirmation.py     # Text and voice interactive permission handlers
│   ├── intent_router.py    # Deterministic zero-latency rule/regex pre-router
│   ├── memory.py           # SQLite persistent memory store & privacy filters
│   ├── tool_router.py      # Bridges LLM structured tool calls to ToolRegistry with permissions
│   └── tool_setup.py       # Centralized tool initialization & registration factory
│
├── ai/
│   ├── circuit_breaker.py  # Thread-safe outage protection state machine
│   ├── ollama_provider.py  # Local Ollama chat and native tool schema adapter
│   └── provider.py         # AIProvider interface + GeminiProvider (REST & SSE)
│
├── tools/
│   ├── base.py             # Base Tool class and sanitized ToolResult model
│   ├── permissions.py      # PermissionLevel enum (SAFE, CONFIRM, DANGEROUS)
│   ├── registry.py         # ToolRegistry catalog and safe execution dispatch
│   ├── applications.py     # App launch tools & allowlist (Notepad, Brave, Code, Calc, etc.)
│   ├── files.py            # Bounded filesystem operations (list, search, move, delete)
│   ├── system.py           # Windows OS & hardware inspection tools (psutil)
│   ├── system_audio.py     # Windows Core Audio COM volume & mute controls
│   ├── system_media.py     # Multimedia playback controls (virtual key events)
│   ├── validation.py       # JSON schema & argument validation guards
│   └── websites.py         # Allowlist HTTPS website opener (YouTube, GitHub, etc.)
│
├── voice/
│   ├── activation.py       # Wake-word detection (AERIS) & active context window
│   ├── response_cleanup.py # Strips markdown, JSON, and code blocks before TTS
│   ├── speech_to_text.py   # SpeechRecognition + PyAudio with device discovery
│   ├── stream_buffer.py    # Natural sentence chunker for streaming speech
│   └── text_to_speech.py   # pyttsx3 offline speech synthesis
│
└── tests/                  # Automated unit, integration, and security test suite
```

---

## Technology Stack

- **Operating System**: Windows 10 / 11
- **Language**: Python 3.11+ (Tested on Python 3.12.0)
- **Speech-to-Text (STT)**: `SpeechRecognition`, `PyAudio`
- **Text-to-Speech (TTS)**: `pyttsx3` (Offline SAPI5 synthesis)
- **Windows Integration**: `comtypes` (Core Audio `IAudioEndpointVolume`), `pywin32` / `ctypes`
- **System Monitoring**: `psutil`
- **Database / Memory**: SQLite 3 (Python standard library `sqlite3`)
- **HTTP / Networking**: `requests`, SSE streaming
- **AI Providers**: Google Gemini API, Local Ollama (`/api/chat`)
- **Test Framework**: `pytest`

---

## Installation & Setup

### 1. Prerequisites
- Windows 10 or 11
- Python 3.11 or 3.12 installed
- Working microphone and speaker/headset (for Voice Mode)

### 2. Create and Activate Virtual Environment

```powershell
# Navigate to the project root
cd D:\A.E.R.I.S\AERIS-phase1\AERIS

# Create virtual environment
python -m venv ..\..\.venv

# Activate virtual environment
..\..\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration (`.env`)

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Edit `.env` to configure your preferred AI provider:

#### Option A: Google Gemini (Cloud)
```dotenv
AI_PROVIDER=gemini
AI_API_KEY=your-gemini-api-key-here
AI_MODEL=gemini-3.5-flash-lite
```

#### Option B: Ollama (Local Offline Model)
```dotenv
AI_PROVIDER=ollama
AI_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=120
OLLAMA_KEEP_ALIVE=24h
OLLAMA_WARMUP=true
```

*Note: Never commit `.env` to version control. It is strictly excluded by `.gitignore`.*

---

## Running A.E.R.I.S.

### Interactive Startup Mode
```powershell
python main.py
```
Prompts you to select:
- `[1] Text mode` — Interactive terminal conversation
- `[2] Voice mode` — Continuous voice-to-voice loop

### Direct Mode Launchers
```powershell
# Launch directly in Text Mode
python main.py --text

# Launch directly in Voice Mode
python main.py --voice
```

### Diagnostic & Hardware Commands
```powershell
# Test AI Provider connectivity and configuration
python main.py --provider-test

# List all detected audio input and output devices
python main.py --devices

# Run local microphone hardware diagnostic
python main.py --audio-test
```

---

## Desktop Tool System & Safety Model

A.E.R.I.S. enforces strict permission boundaries through the **ToolRouter**:

| Tool Name | Permission Level | Description |
| :--- | :---: | :--- |
| `system_info` | `SAFE` | Reports live CPU, RAM, disk, battery, and Windows OS details. |
| `system_audio` | `SAFE` | Adjusts master volume (up/down) and toggles mute/unmute. |
| `system_media` | `SAFE` | Controls multimedia playback (play/pause, next track, prev track). |
| `list_files` | `SAFE` | Lists directory contents with item limits and traversal guards. |
| `search_files` | `SAFE` | Searches files by pattern matching (capped at 1,000 results). |
| `get_file_info` | `SAFE` | Retrieves file metadata (size, timestamps, MIME hints). |
| `open_website` | `CONFIRM` | Opens allowlisted HTTPS domains (YouTube, Google, GitHub, etc.). |
| `open_application` | `CONFIRM` | Launches allowlisted Windows apps (Notepad, Brave, VS Code, Calc). |
| `create_folder` | `CONFIRM` | Creates directories after explicit user consent. |
| `move_file` | `CONFIRM` | Relocates files with collision protection. |
| `rename_file` | `CONFIRM` | Renames files in-place without shell invocation. |
| `delete_file` | `DANGEROUS` | Permanently deletes files; requires explicit confirmation. |

When approval is required:
- In **Text Mode**: Prompts `Approve? (yes/no): ` in the terminal.
- In **Voice Mode**: Speaks the confirmation prompt and listens for spoken approval (`"yes"`, `"do it"`, `"sure"`, `"cancel"`).

---

## Local Intent Router

To optimize responsiveness and avoid unnecessary cloud latency, A.E.R.I.S. includes a local deterministic **IntentRouter** that intercepts common commands before querying the AI model:

| Spoken or Typed Command | Intercepted Action |
| :--- | :--- |
| `"turn up the volume"`, `"louder"`, `"volume down"` | Adjusts Windows master volume |
| `"mute audio"`, `"unmute sound"` | Toggles system mute via Windows Core Audio |
| `"pause music"`, `"resume song"`, `"next track"` | Dispatches Windows media control keys |
| `"open brave"`, `"launch notepad"`, `"start calculator"` | Launches approved desktop applications |
| `"open github"`, `"open youtube"`, `"visit chatgpt"` | Opens canonical allowlisted HTTPS websites |
| `"what is my cpu usage?"`, `"check ram"` | Retrieves live system hardware metrics |
| `"stop"`, `"be quiet"`, `"cancel"` | Immediately interrupts active voice speech |
| `"switch to text mode"`, `"exit"`, `"goodbye"` | Local session state navigation |

Conceptual questions (e.g. *"What is a CPU?"*, *"How does YouTube work?"*) are automatically excluded from local routing and forwarded to the conversational AI.

---

## Automated Test Suite

A.E.R.I.S. includes a comprehensive test suite covering all modules, security boundaries, and edge cases:

```powershell
# Run the complete test suite
python -m pytest -q

# Run focused tests for desktop tools
python -m pytest tests/test_system_audio.py tests/test_system_media.py tests/test_applications.py tests/test_websites.py -q

# Run provider reliability tests
python -m pytest tests/test_provider.py tests/test_ollama_provider.py tests/test_provider_reliability.py -q
```

**Verified Test Baseline:**
- **686 passed, 1 skipped, 2 warnings, 2 subtests passed** in ~31 seconds.
- *Note on skipped test*: Windows un-elevated execution skips symlink creation test, as expected on non-admin environments.

---

## Current Limitations

- **Windows Exclusive**: Relies on Windows Core Audio COM APIs, Win32 multimedia virtual key events, and Windows application path resolution.
- **Microphone Arbitration**: Sequential voice scheduling requires the microphone to close while TTS is speaking to prevent acoustic feedback loops.
- **Allowlist Scope**: Application and website launching are governed by strict allowlists; arbitrary URL execution or custom executable paths require manual allowlist addition.

---

## Roadmap

- [x] Phase 1 — Core terminal assistant & AI provider abstraction
- [x] Phase 2 — Voice input and output (SpeechRecognition & pyttsx3)
- [x] Phase 3 — Tool system with permissions (applications, files, system info)
- [x] Phase 4 — Real-time continuous voice assistant loop with microphone fallback
- [x] Phase 5 — Intelligent Assistant Core (conversation memory, system prompt, tool sanitization)
- [x] Phase 6 — True voice-to-voice assistant (local command interception, timeout protection)
- [x] Phase 7 — Low-latency voice performance and instrumentation
- [x] Phase 8 — Wake-word activation (`aeris`), active voice context, barge-in interruption
- [x] Phase 9 — Gemini SSE streaming, sentence buffering, and queued voice output
- [x] Phase 10A — Windows Master Audio Controls (volume up/down, mute/unmute via Core Audio)
- [x] Phase 10B — Windows Media Playback Controls (play/pause, next/prev track)
- [x] Phase 10C — Brave Browser launch support & application allowlist expansion
- [x] Phase 10D — Dedicated allowlisted HTTPS website opener & scheme validation
- [x] Phase 10E — Environment & runtime stabilization
- [ ] Phase 10F — Window & desktop management (inspect, focus, minimize, snap active windows)
- [ ] Phase 11 — Native system tray daemon & floating companion HUD
- [ ] Phase 12 — Screen perception & vision integration via multimodal AI
