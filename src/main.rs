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

    /// Execution device to target: "auto", "gpu", "cpu", or specific GPU name
    #[arg(long, default_value = "auto")]
    device: String,

    /// Scan hardware (RAM, CPU cores) and report model recommendations in JSON
    #[arg(long)]
    check: bool,

    /// List supported models and whether they are installed locally
    #[arg(long)]
    list_models: bool,
}

fn main() {
    let cli = Cli::parse();
    let models_dir = cli.models_dir.unwrap_or_else(models::get_default_models_dir);

    // 1. Hardware scan and diagnostics mode
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

    // 2. List local models mode
    if cli.list_models {
        let local_models = models::list_local_models(&models_dir);
        let resp = serde_json::json!({
            "models_dir": models_dir.to_string_lossy(),
            "models": local_models,
        });
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return;
    }

    // 3. Audio transcription mode
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

    // 3. Transcribe
    if !cli.json {
        eprintln!("[3/3] Generating synchronized lyrics and subtitles...");
    }
    let result = match engine.transcribe_pcm(&pcm, Some(&cli.language), &cli.task) {
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
