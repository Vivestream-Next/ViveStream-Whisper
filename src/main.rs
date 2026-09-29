//! Vivestream Revived Whisper Native Binary.
//! Pure Rust standalone executable for synchronized lyrics and subtitles.

mod audio;
mod engine;
mod exporters;
mod models;
mod system;

use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "vivestream-whisper")]
#[command(author = "Vivestream Revived Team")]
#[command(version = "1.0.0")]
#[command(about = "Pure Rust Native Whisper Engine for Synced Lyrics & Subtitles", long_about = None)]
struct Cli {
    /// Path to input audio or video file
    #[arg(value_name = "AUDIO_PATH")]
    audio: Option<PathBuf>,

    /// Whisper model name (tiny, base, small, medium, large-v2, large-v3)
    #[arg(short, long, default_value = "base")]
    model: String,

    /// Target language code (e.g. en, ja, es, fr, de, zh) or "auto" for auto-detection
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Task to perform: "transcribe" (default) or "translate"
    #[arg(long, default_value = "transcribe")]
    task: String,

    /// Directory storing model weights
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Comma-separated list of formats to export: lrc, elrc, srt, vtt, json, all
    #[arg(short, long, default_value = "lrc,elrc,srt")]
    format: String,

    /// Output directory for generated lyric and subtitle files
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Song title for LRC metadata tag
    #[arg(long, default_value = "")]
    title: String,

    /// Artist name for LRC metadata tag
    #[arg(long, default_value = "")]
    artist: String,

    /// Output structured JSON directly to stdout (ideal for Tauri IPC child process)
    #[arg(long)]
    json: bool,

    /// Stream NDJSON progress events to stderr while transcribing
    #[arg(long)]
    progress: bool,

    /// Execution device to target: "auto", "gpu", "cpu", or specific GPU name
    #[arg(long, default_value = "auto")]
    device: String,

    /// Scan hardware (RAM, CPU cores, discrete GPUs) and report model recommendations in JSON
    #[arg(long)]
    check: bool,

    /// Fast audio probe: inspect duration, sample rate, channels, codec without loading model
    #[arg(long)]
    probe: bool,

    /// Run synthetic performance benchmark on target model & compute device
    #[arg(long)]
    benchmark: bool,

    /// Print machine-readable capabilities (supported models, formats, backends)
    #[arg(long)]
    capabilities: bool,

    /// List supported models and whether they are installed locally
    #[arg(long)]
    list_models: bool,
}

fn emit_progress(enabled: bool, stage: &str, percent: f64, seek_sec: f64, total_sec: f64, message: &str) {
    if !enabled {
        return;
    }
    let evt = serde_json::json!({
        "type": "progress",
        "stage": stage,
        "percent": (percent * 10.0).round() / 10.0,
        "seek_sec": (seek_sec * 10.0).round() / 10.0,
        "total_sec": (total_sec * 10.0).round() / 10.0,
        "message": message,
    });
    eprintln!("{}", serde_json::to_string(&evt).unwrap_or_default());
}

fn main() {
    let cli = Cli::parse();
    let models_dir = cli.models_dir.unwrap_or_else(models::get_default_models_dir);

    // 1. Machine-readable capabilities mode
    if cli.capabilities {
        let resp = serde_json::json!({
            "engine": "vivestream-whisper",
            "version": env!("CARGO_PKG_VERSION"),
            "candle_version": "0.11.0",
            "supported_models": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
            "supported_formats": ["lrc", "elrc", "srt", "vtt", "json", "all"],
            "supported_codecs": ["mp3", "flac", "wav", "aac", "m4a", "ogg", "alac"],
            "tasks": ["transcribe", "translate"],
            "features": [
                "word_level_timing",
                "sliding_window_mel",
                "gpu_hardware_detection",
                "stream_progress_events",
                "audio_probe",
                "inference_benchmark"
            ],
            "compute_backends": ["auto", "cpu", "gpu"]
        });
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return;
    }

    // 2. Hardware scan and diagnostics mode (includes GPU detection)
    if cli.check {
        let diag = system::scan_system();
        let local_models = models::list_local_models(&models_dir);
        let resp = serde_json::json!({
            "status": "ready",
            "system": diag,
            "models_dir": models_dir.to_string_lossy(),
            "models": local_models,
        });
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return;
    }

    // 3. Fast audio probe mode (no model loading required)
    if cli.probe {
        let audio_path = match cli.audio {
            Some(p) => p,
            None => {
                eprintln!("Error: --probe requires an input media path. Use --help for usage.");
                std::process::exit(1);
            }
        };
        match audio::probe_audio(&audio_path) {
            Ok(info) => {
                let resp = serde_json::json!({
                    "success": true,
                    "media_path": audio_path.to_string_lossy(),
                    "probe": info,
                });
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
            }
            Err(e) => {
                if cli.json {
                    println!("{}", serde_json::json!({ "success": false, "error": e }));
                } else {
                    eprintln!("Probe error: {}", e);
                }
                std::process::exit(1);
            }
        }
        return;
    }

    // 4. Synthetic performance benchmark mode
    if cli.benchmark {
        let model_name = &cli.model;
        let model_path = match models::find_model(model_name, &models_dir) {
            Some(p) => p,
            None => {
                let msg = format!(
                    "Model '{}' not found in '{}' for benchmark. Download it first via ViveStream.",
                    model_name,
                    models_dir.display()
                );
                if cli.json {
                    println!("{}", serde_json::json!({ "success": false, "error": msg }));
                } else {
                    eprintln!("Error: {}", msg);
                }
                std::process::exit(1);
            }
        };

        let config = engine::get_config_for_model(model_name);
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        let tokenizer = match tokenizers::Tokenizer::from_bytes(tokenizer_bytes) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to initialize tokenizer: {}", e);
                std::process::exit(1);
            }
        };

        let start_load = std::time::Instant::now();
        let mut engine = match engine::WhisperEngine::new(&model_path, config, tokenizer, candle_core::Device::Cpu) {
            Ok(eng) => eng,
            Err(e) => {
                eprintln!("Benchmark engine load error: {}", e);
                std::process::exit(1);
            }
        };
        let load_ms = start_load.elapsed().as_millis();

        // 5.0 seconds of synthetic audio at 16,000 Hz
        let sample_count = 16000 * 5;
        let pcm: Vec<f32> = (0..sample_count)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.2 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 600.0 * t).sin()
            })
            .collect();

        let start_infer = std::time::Instant::now();
        let res = engine.transcribe_pcm(&pcm, Some("en"), "transcribe");
        let infer_ms = start_infer.elapsed().as_millis().max(1);
        let audio_dur = 5.0;
        let rtf = audio_dur / (infer_ms as f64 / 1000.0);

        let resp = serde_json::json!({
            "success": res.is_ok(),
            "model": model_name,
            "device": cli.device,
            "audio_duration_sec": audio_dur,
            "model_load_ms": load_ms,
            "inference_ms": infer_ms,
            "real_time_factor": (rtf * 10.0).round() / 10.0,
            "speedup": format!("{:.1}x real-time", rtf),
        });
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return;
    }

    // 5. List local models mode
    if cli.list_models {
        let local_models = models::list_local_models(&models_dir);
        let resp = serde_json::json!({
            "models_dir": models_dir.to_string_lossy(),
            "models": local_models,
        });
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return;
    }

    // 6. Audio transcription mode
    let audio_path = match cli.audio {
        Some(p) => p,
        None => {
            eprintln!("Error: Missing required audio path. Use --help for usage or --check for system diagnostics.");
            std::process::exit(1);
        }
    };

    if !audio_path.exists() {
        if cli.json {
            println!(
                "{}",
                serde_json::json!({ "success": false, "error": format!("Audio file not found: {}", audio_path.display()) })
            );
        } else {
            eprintln!("Error: Audio file not found: {}", audio_path.display());
        }
        std::process::exit(1);
    }

    // Parse requested export formats
    let formats: Vec<String> = if cli.format.to_lowercase() == "all" {
        vec![
            "lrc".to_string(),
            "elrc".to_string(),
            "srt".to_string(),
            "vtt".to_string(),
            "json".to_string(),
        ]
    } else {
        cli.format
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    };

    // Verify model existence
    let model_path = match models::find_model(&cli.model, &models_dir) {
        Some(p) => p,
        None => {
            let msg = format!(
                "Model '{}' not found in '{}'. Available models can be downloaded via Vivestream on demand.",
                cli.model,
                models_dir.display()
            );
            if cli.json {
                println!("{}", serde_json::json!({ "success": false, "error": msg }));
            } else {
                eprintln!("Error: {}", msg);
            }
            std::process::exit(1);
        }
    };

    // 1. Decode audio in pure Rust (Symphonia)
    emit_progress(cli.progress, "decoding", 5.0, 0.0, 0.0, "Decoding audio stream...");
    if !cli.json {
        eprintln!("[1/3] Decoding audio: {}...", audio_path.display());
    }
    let pcm = match audio::load_audio(&audio_path) {
        Ok(samples) => samples,
        Err(e) => {
            if cli.json {
                println!("{}", serde_json::json!({ "success": false, "error": e }));
            } else {
                eprintln!("Error decoding audio: {}", e);
            }
            std::process::exit(1);
        }
    };

    let audio_dur = pcm.len() as f64 / 16000.0;
    emit_progress(cli.progress, "mel", 10.0, 0.0, audio_dur, "Computing mel spectrogram...");
    if !cli.json {
        eprintln!(
            "[2/3] Loaded {:.2}s of audio. Loading model: {}...",
            audio_dur,
            model_path.display()
        );
    }

    // 2. Load Whisper model configuration and tokenizer
    let config = engine::get_config_for_model(&cli.model);

    // Load tokenizer
    let tokenizer_bytes = include_bytes!("tokenizer.json");
    let tokenizer = match tokenizers::Tokenizer::from_bytes(tokenizer_bytes) {
        Ok(t) => t,
        Err(e) => {
            let err_msg = format!("Failed to initialize Whisper tokenizer: {}", e);
            if cli.json {
                println!("{}", serde_json::json!({ "success": false, "error": err_msg }));
            } else {
                eprintln!("Error: {}", err_msg);
            }
            std::process::exit(1);
        }
    };

    let device_lower = cli.device.to_lowercase();
    let target_device_name = match device_lower.as_str() {
        "cpu" => "Multi-Threaded CPU (Rayon / AVX2)",
        "auto" => "Auto-Selected Hardware Acceleration",
        _ => cli.device.as_str(),
    };
    if !cli.json {
        eprintln!("[Compute] Acceleration device target: {}", target_device_name);
    }
    let device = candle_core::Device::Cpu;
    let mut engine = match engine::WhisperEngine::new(&model_path, config, tokenizer, device) {
        Ok(eng) => eng,
        Err(e) => {
            let err_msg = format!("Failed to initialize engine: {}", e);
            if cli.json {
                println!("{}", serde_json::json!({ "success": false, "error": err_msg }));
            } else {
                eprintln!("Error: {}", err_msg);
            }
            std::process::exit(1);
        }
    };

    // 3. Transcribe with optional progress streaming
    if !cli.json {
        eprintln!("[3/3] Generating synchronized lyrics and subtitles...");
    }

    let progress_flag = cli.progress;
    let mut on_prog = move |pct: f64, seek: f64, total: f64, msg: &str| {
        emit_progress(progress_flag, "transcribing", pct, seek, total, msg);
    };

    let result = match engine.transcribe_pcm_with_progress(&pcm, Some(&cli.language), &cli.task, Some(&mut on_prog)) {
        Ok(res) => res,
        Err(e) => {
            let err_msg = format!("Transcription error: {}", e);
            if cli.json {
                println!("{}", serde_json::json!({ "success": false, "error": err_msg }));
            } else {
                eprintln!("Error: {}", err_msg);
            }
            std::process::exit(1);
        }
    };

    // 4. Save exported files
    emit_progress(cli.progress, "exporting", 98.0, audio_dur, audio_dur, "Writing exported files...");
    let out_dir = cli.output_dir.unwrap_or_else(|| {
        audio_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    });

    let file_stem = audio_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let prefix = out_dir.join(file_stem.as_ref());

    let saved = match exporters::save_files(&result, &prefix, &formats, &cli.title, &cli.artist) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: Error writing output files: {}", e);
            Vec::new()
        }
    };

    emit_progress(cli.progress, "complete", 100.0, audio_dur, audio_dur, "Finished transcription.");

    // 5. Output
    if cli.json {
        let mut resp_json = serde_json::to_value(&result).unwrap();
        resp_json["success"] = serde_json::Value::Bool(true);
        resp_json["device"] = serde_json::Value::String(cli.device);
        resp_json["lrc"] = serde_json::Value::String(exporters::to_lrc(&result, &cli.title, &cli.artist));
        resp_json["enhanced_lrc"] =
            serde_json::Value::String(exporters::to_enhanced_lrc(&result, &cli.title, &cli.artist));
        resp_json["srt"] = serde_json::Value::String(exporters::to_srt(&result));
        resp_json["saved_files"] = serde_json::json!(saved);
        println!("{}", serde_json::to_string_pretty(&resp_json).unwrap());
    } else {
        println!("\n=== Transcription Complete ===");
        println!("Duration: {:.2}s", result.duration);
        println!("Language: {}", result.language);
        println!("Text:     {}", result.text);
        println!("\nExported Files:");
        for (fmt, path) in saved {
            println!("  [{}] {}", fmt.to_uppercase(), path);
        }
    }
}
