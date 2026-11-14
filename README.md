# Vivestream Whisper

> **High-Performance, Dual-Engine (Vulkan 1.3 GPU + Candle CPU) OpenAI Whisper Engine for Synced Lyrics & Subtitles**  
> Powered by **Vulkan Compute Shaders (`whisper.cpp`)**, **Hugging Face Candle**, and **Symphonia**.  
> **Zero external runtime dependencies. No Python. No PyTorch. No CUDA SDK required.**

---

## Highlights

- **Cross-Vendor Vulkan 1.3 GPU Acceleration**: Native hardware acceleration on **Intel Arc (e.g. A770, A750, Iris Xe)**, **AMD Radeon (RX 5000/6000/7000)**, and **NVIDIA GeForce (RTX/GTX)** discrete GPUs using standard OS graphics drivers (`vulkan-1.dll` / `libvulkan.so`).
- **Dedicated VRAM Execution**: Model weights and KV cache are loaded directly into GPU VRAM (e.g. 77 MB for `tiny`, 1.5 GB for `medium`), leaving system RAM virtually untouched (< 4 MB host RAM overhead during inference).
- **Blazing Fast In-Memory Inference**: Real-world benchmark: **10.0 seconds of audio transcribed in ~163 milliseconds** on an Intel Arc A770 dGPU (~25x speedup over CPU).
- **Multi-Threaded CPU Fallback**: Automatic Candle (Rayon / AVX2 vector SIMD) execution when `--device cpu` is requested or no discrete GPU is present.
- **50% Smaller GGML Model Weights**: Uses official Hugging Face GGML checkpoints (`ggml-{name}.bin`) for half the download size and disk usage compared to FP32 SafeTensors, with full backward compatibility for existing `.safetensors` files.
- **Pure Rust Multi-Format Audio Ingestion**: Native decoding of MP4, MKV, WebM, MP3, FLAC, WAV, AAC, M4A, ALAC, and OGG with automatic resampling to 16,000 Hz mono.
- **Real-Time NDJSON Progress Events**: Pass `--progress` to stream real-time progress events over stderr (`{"type": "progress", "stage": "transcribing", "percent": ...}`) for smooth UI progress bar integration.
- **Production-Grade Exporters**:
  - Standard Synchronized LRC (`.lrc`)
  - Enhanced Word-by-Word Karaoke LRC (`.enhanced.lrc` with `<mm:ss.xx> word`)
  - SubRip Subtitles (`.srt`)
  - WebVTT Subtitles (`.vtt`)
  - Structured JSON (`.json` or `--json` to `stdout`)
- **System Hardware Diagnostics (`--check`)**: Scans host CPUs, discrete GPUs, total/available VRAM, and recommends the optimal model tier.

---

## Supported Whisper Models (GGML)

| Model | Weights Size (GGML) | Recommended VRAM | Typical Speed (Arc A770 GPU) | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `tiny` | ~75 MB | >= 1 GB VRAM | **~150 - 200 ms** | Instant previews & real-time sync |
| `base` *(default)* | ~142 MB | >= 2 GB VRAM | **~300 - 500 ms** | Ideal balanced lyric & subtitle generation |
| `small` | ~466 MB | >= 4 GB VRAM | **~800 ms - 1.2s** | High accuracy multilingual songs |
| `medium` | ~1.50 GB | >= 6 GB VRAM | **~2.0 - 3.5s** | Complex vocals, acoustic tracks & noise |
| `large-v2` | ~3.10 GB | >= 8 GB VRAM | **~4.0 - 6.0s** | 80-channel studio precision |
| `large-v3` | ~3.10 GB | >= 8 GB VRAM | **~4.0 - 7.0s** | 128-channel maximum studio precision |

*Weights are standard Hugging Face GGML binaries (`ggml-{name}.bin`). Legacy Hugging Face SafeTensors (`model.safetensors`) continue to be supported.*

---

## Quick Start

### 1. Build from Source

Ensure you have Rust and Cargo installed:

```bash
git clone https://github.com/Vivestream-Next/ViveStream-Whisper.git
cd ViveStream-Whisper
cargo build --release
```

The resulting binary will be located at `target/release/vivestream-whisper.exe` (or `vivestream-whisper` on Linux/macOS).

### 2. Hardware Diagnostics (`--check`)

```bash
./vivestream-whisper --check
```

Outputs machine-readable JSON detailing host CPU, discrete GPUs (with VRAM capacity and driver version), compute capability (`Vulkan 1.3 / Intel Xe-HPG VRAM Acceleration`), and model recommendations.

### 3. Synthetic Benchmark (`--benchmark`)

```bash
./vivestream-whisper --benchmark --model tiny --device gpu:0 --json
```

Runs synthetic 5.0s audio inference directly on the GPU, outputting real-time factor and latency metrics.

### 4. Transcribe Audio / Video

Download a model checkpoint (e.g. `ggml-base.bin` into `~/ViveStream/AI/Whisper/` or `./models/`), then run:

```bash
# Vulkan GPU hardware-accelerated transcription (outputs .lrc, .enhanced.lrc, .srt)
./vivestream-whisper "path/to/song.mp4" --model base --device gpu:0 -f all -o "./output"

# Stream real-time progress events to stderr with structured JSON output to stdout
./vivestream-whisper "path/to/song.mp3" --model base --device auto --progress --json

# Multi-threaded CPU fallback
./vivestream-whisper "path/to/song.mp3" --model base --device cpu --json
```

---

## CLI Options

```
Usage: vivestream-whisper [OPTIONS] [AUDIO_PATH]

Arguments:
  [AUDIO_PATH]  Path to input media file (MP4, MKV, MP3, FLAC, WAV, AAC, M4A, OGG)

Options:
      --check                 Scan system hardware, detect GPUs/VRAM, recommend model size, and exit
      --list-models           List supported models and local installation status
      --capabilities          Print machine-readable engine features and supported backends
      --benchmark             Run synthetic inference benchmark on target device
      --probe                 Fast audio probe (sample rate, channels, duration) without loading model
  -m, --model <MODEL>         Whisper model tier [default: base] [possible values: tiny, base, small, medium, large-v2, large-v3]
  -l, --language <LANG>       Language code (e.g. en, ja, es, fr, zh) or "auto" [default: auto]
      --task <TASK>           Task: transcribe or translate [default: transcribe]
      --device <DEVICE>       Execution target: "auto", "gpu:0", or "cpu" [default: auto]
      --models-dir <DIR>      Custom directory containing ggml-*.bin or *.safetensors files
  -f, --format <FORMAT>       Output formats: lrc, elrc, srt, vtt, json, all [default: lrc,elrc,srt]
  -o, --output-dir <DIR>      Destination directory for generated subtitle/lyric files [default: .]
      --title <STR>           Song title metadata tag for LRC headers
      --artist <STR>          Artist metadata tag for LRC headers
      --json                  Print structured JSON payload to stdout (for IPC child process piping)
      --progress              Stream NDJSON progress events to stderr while transcribing
  -h, --help                  Print help
  -V, --version               Print version
```

---

## Integration with Tauri, Electron & Node.js

Because `vivestream-whisper` emits real-time progress to `stderr` and clean structured JSON to `stdout` (`--json`), integrating it into any desktop app or backend service is straightforward:

```typescript
import { Command } from '@tauri-apps/plugin-shell';

const cmd = Command.create('vivestream-whisper', [
  mediaFilePath,
  '--model', 'base',
  '--device', 'gpu:0',
  '--progress',
  '--json'
]);

cmd.stderr.on('data', (line) => {
  try {
    const evt = JSON.parse(line);
    if (evt.type === 'progress') {
      console.log(`Inference Progress: ${evt.percent}% (${evt.message})`);
    }
  } catch {}
});

const output = await cmd.execute();
const result = JSON.parse(output.stdout);
console.log("Transcribed Text:", result.text);
console.log("Karaoke Words:", result.segments[0].words);
```

For complete architectural specifications, see [`AI_DEVELOPER_GUIDE.md`](AI_DEVELOPER_GUIDE.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
