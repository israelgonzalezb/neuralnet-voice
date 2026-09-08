# voice-prompt

A 4MB native Windows CLI tool for instant voice-to-prompt compilation.

No Python virtualenvs, no CUDA dependencies, zero disk I/O. Captures audio directly from physical hardware via WASAPI, transcribes sub-second with Google's `gemini-3.5-transcribe` model, biases for technical vocabulary, optionally compiles rambling speech into prompt architecture via `gemini-3.5-flash-lite`, and drops the result directly into your clipboard.

```
[WASAPI Capture] -> [16kHz In-Memory Resample] -> [Gemini 3.5 Transcribe] -> [Flash-Lite Refine] -> [Clipboard]
```

---

## Features

- **Zero Disk Audio I/O**: Direct WASAPI shared-mode hardware capture. Audio downmixing and resampling to 16,000 Hz Mono I16 happen in RAM.
- **Custom Vocabulary Biasing**: Injects domain vocabulary at the API layer so technical terms, code syntax, and shell tools (`Rust`, `WASAPI`, `Nushell`, `JSONL`, `ONNX`) aren't mangled.
- **Prompt Architecture Refinement (`-r`)**: Converts spoken thoughts into dense, structured, actionable instructions formatted for AI code assistants and LLMs.
- **Model Auditability & Fallback**: Handles transient 503 capacity spikes with exponential backoff and dynamic fallback to `gemini-3.5-flash`, explicitly logging the resolving model.
- **Resilient Clipboard Injection (`-c`)**: Threaded clipboard injection with automatic Win32 lock retry and fallback to `clip.exe`.
- **Telemetry Ledger**: Appends structured JSON records to `%LOCALAPPDATA%\neuralnet\voice\history.jsonl` tracking latency breakdown (`t_record`, `t_transcribe`, `t_refine`, `t_total`), model resolution, and raw transcripts.

---

## Installation

### Prebuilt Binary (Windows x64)
Download `voice-prompt.exe` from [GitHub Releases](https://github.com/israelgonzalezb/neuralnet-voice/releases) and place it in your `PATH` (e.g., `~/.cargo/bin/` or `C:\Windows\System32`).

### From Source
```powershell
cargo install --git https://github.com/israelgonzalezb/neuralnet-voice
```

---

## Configuration

Set your Gemini API key in your environment:

```powershell
# PowerShell
$env:GEMINI_API_KEY = "AIzaSy..."
```

```nushell
# Nushell
$env.GEMINI_API_KEY = "AIzaSy..."
```

*If `$env:GEMINI_API_KEY` is not set, `voice-prompt` automatically checks `$env:APPDATA/nushell/secrets.nu`.*

---

## Usage

```powershell
# Interactive mode (Press Enter to stop recording and transcribe):
voice-prompt

# 5-second capture, compile into prompt architecture, copy directly to clipboard:
voice-prompt -d 5 -r -c

# Short flags vector:
voice-prompt -r -c

# Specific audio device override:
voice-prompt --device "Yeti"
```

### CLI Flags

| Flag | Description |
| :--- | :--- |
| `-r, --refine` | Refines raw spoken thought into dense prompt architecture |
| `-c, --copy` | Injects raw transcript or refined prompt into Windows clipboard |
| `-d, --duration <secs>` | Fixed recording duration in seconds (default: interactive Enter) |
| `--device <string>` | DirectShow / WASAPI audio device name substring override |

---

## License

MIT
