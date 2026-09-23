# Vivestream Whisper

> **High-Performance, Pure Rust OpenAI Whisper Engine for Synced Lyrics & Subtitles**  
> Powered by [Hugging Face Candle](https://github.com/huggingface/candle) and [Symphonia](https://github.com/pdeljanov/Symphonia).  
> **Zero external runtime dependencies. No Python. No PyTorch. No CUDA/C++ SDK required.**

---

## Highlights

- **100% Pure Rust**: Embedded multilingual BPE tokenizer and 80-channel log-mel filterbank inside a **~7.8 MB standalone executable**.
- **Arbitrary Length Audio Support**: Intelligent 30-second sliding mel spectrogram windowing (`mel.narrow(2, seek, segment_size)`) allowing seamless transcription of full-length songs and podcasts without embedding bounds errors.
- **Word-Level Timing**: Accurate word interval alignments and natural sentence token spacing.
- **Comprehensive Exporters**:
  - Standard Synchronized LRC (`.lrc`)
  - Enhanced Word-by-Word Karaoke LRC (`.enhanced.lrc` with `<mm:ss.xx> word`)
  - SubRip Subtitles (`.srt`)
  - WebVTT Subtitles (`.vtt`)
  - Structured JSON (`.json` or `--json` to `stdout`)
- **System Hardware Diagnostics (`--check`)**: Scans CPU cores and system RAM to recommend the ideal model tier.
- **SafeTensors Management (`--list-models`)**: Discovers local models and resolves official Hugging Face download URLs for on-demand downloading.

---

## Supported Whisper Models

| Model | Weights Size (FP32) | Recommended RAM | Typical Speed (CPU) | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `tiny` | ~151 MB | >= 1 GB | ~1 - 2s | Ultra-fast previews & low-end devices |
| `base` *(default)* | ~290 MB | >= 2 GB | ~2 - 4s | Ideal balanced music lyric generation |
| `small` | ~967 MB | >= 4 GB | ~4 - 8s | High accuracy multilingual transcription |
| `medium` | ~3.06 GB | >= 6 GB | ~8 - 15s | Complex vocal arrangements & background noise |
| `large-v2` | ~6.18 GB | >= 8 GB | ~15 - 20s | 80-channel studio-grade precision |
| `large-v3` | ~6.18 GB | >= 8 GB | ~15 - 25s | 128-channel maximum studio precision |

*All weights are standard Hugging Face SafeTensors (`model.safetensors`).*

---

## Quick Start

### 1. Build from Source

Ensure you have Rust and Cargo installed:

```bash
git clone https://github.com/your-username/vivestream-whisper.git
cd vivestream-whisper
cargo build --release
```

The resulting binary will be located at `target/release/vivestream-whisper` (or `.exe` on Windows).

### 2. Check System Hardware & Recommended Model

```bash
./vivestream-whisper --check
```

Outputs JSON detailing CPU model, cores, total/available RAM, and model recommendations.

### 3. List & Resolve Models

```bash
./vivestream-whisper --list-models
```

### 4. Transcribe Audio

Download a model checkpoint (e.g. `base.safetensors` from Hugging Face into a `models/` directory), then run:

```bash
# Generate all lyric and subtitle formats
./vivestream-whisper "path/to/song.mp3" --model base --models-dir "./models" -f all -o "./output"

# Multilingual transcription (or auto-detect language)
./vivestream-whisper "path/to/song.mp3" --model base --language ja -f all

# Output structured JSON directly to stdout (ideal for IPC pipes, Tauri, or Electron)
./vivestream-whisper "path/to/song.mp3" --model base --models-dir "./models" --json
```

---

## CLI Options

```
Usage: vivestream-whisper [OPTIONS] [AUDIO_PATH]

Arguments:
  [AUDIO_PATH]  Path to input audio file (MP3, FLAC, WAV, AAC, M4A, OGG)

Options:
      --check                 Scan system hardware, recommend model size, and exit
      --list-models           List supported models and installation status
  -m, --model <MODEL>         Whisper model tier [default: base] [possible values: tiny, base, small, medium, large-v2, large-v3]
  -l, --language <LANG>       Language code (e.g. en, ja, es, fr) or "auto" [default: auto]
      --task <TASK>           Task: transcribe or translate [default: transcribe]
      --models-dir <DIR>      Custom directory containing .safetensors files
  -f, --format <FORMAT>       Output formats: lrc, elrc, srt, vtt, json, all [default: lrc,elrc,srt]
  -o, --output-dir <DIR>      Destination directory for generated files [default: .]
      --title <STR>           Title metadata for LRC headers
      --artist <STR>          Artist metadata for LRC headers
      --json                  Print structured JSON payload to stdout
  -h, --help                  Print help
  -V, --version               Print version
```


---

## Integration with Tauri, Electron & Node.js

Because `vivestream-whisper` can output pure JSON to `stdout` (`--json`), integrating it into any desktop app or backend service is effortless:

```typescript
import { Command } from '@tauri-apps/plugin-shell';

const output = await Command.create('vivestream-whisper', [
  audioFilePath,
  '--model', 'base',
  '--models-dir', modelsDir,
  '--json'
]).execute();

const lyrics = JSON.parse(output.stdout);
console.log("Transcribed text:", lyrics.text);
console.log("Karaoke words:", lyrics.segments[0].words);
```

For full details on the IPC API and Tauri Rust backend integration, see [`AI_DEVELOPER_GUIDE.md`](AI_DEVELOPER_GUIDE.md).

---

## CI / CD

The included GitHub Actions workflow (`.github/workflows/release.yml`) builds cross-platform release binaries for:
- Windows (`x86_64-pc-windows-msvc`)
- Linux (`x86_64-unknown-linux-gnu`)
- macOS ARM64 (`aarch64-apple-darwin`)

Every git tag push (e.g. `git tag v1.0.0 && git push origin v1.0.0`) automatically creates a GitHub Release with the bundled executables.

---

## License

This project is licensed under the [MIT License](LICENSE).
