//! Model directory manager and Hugging Face SafeTensors resolver.

use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: &'static str,
    pub repo_id: &'static str,
    pub file_name: &'static str,
    pub size_mb: usize,
    pub is_multilingual: bool,
}

pub const SUPPORTED_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "tiny",
        repo_id: "openai/whisper-tiny",
        file_name: "model.safetensors",
        size_mb: 151,
        is_multilingual: true,
    },
    ModelInfo {
        name: "base",
        repo_id: "openai/whisper-base",
        file_name: "model.safetensors",
        size_mb: 290,
        is_multilingual: true,
    },
    ModelInfo {
        name: "small",
        repo_id: "openai/whisper-small",
        file_name: "model.safetensors",
        size_mb: 967,
        is_multilingual: true,
    },
    ModelInfo {
        name: "medium",
        repo_id: "openai/whisper-medium",
        file_name: "model.safetensors",
        size_mb: 3060,
        is_multilingual: true,
    },
    ModelInfo {
        name: "large-v2",
        repo_id: "openai/whisper-large-v2",
        file_name: "model.safetensors",
        size_mb: 6180,
        is_multilingual: true,
    },
    ModelInfo {
        name: "large-v3",
        repo_id: "openai/whisper-large-v3",
        file_name: "model.safetensors",
        size_mb: 6180,
        is_multilingual: true,
    },
];


pub fn get_default_models_dir() -> PathBuf {
    // 1. Check WHISPER_MODELS_DIR env var
    if let Ok(dir) = std::env::var("WHISPER_MODELS_DIR") {
        return PathBuf::from(dir);
    }

    // 2. Check ~/ViveStream/AI/Whisper (standard ViveStream directory)
    if let Some(user_home) = dirs_home_dir() {
        let p = user_home.join("ViveStream").join("AI").join("Whisper");
        if p.exists() {
            return p;
        }
    }

    // 3. Check local ./models folder
    let local_models = PathBuf::from("models");
    if local_models.exists() {
        return local_models;
    }

    // 4. Default to ~/ViveStream/AI/Whisper
    if let Some(user_home) = dirs_home_dir() {
        return user_home.join("ViveStream").join("AI").join("Whisper");
    }

    // 5. Fallback to AppData/Local/Vivestream/whisper/models
    if let Some(mut base) = dirs_local_data() {
        base.push("VivestreamRevived");
        base.push("whisper");
        base.push("models");
        return base;
    }

    PathBuf::from("models")
}

fn dirs_home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
}

fn dirs_local_data() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
}

pub fn find_model(model_name: &str, base_dir: &Path) -> Option<PathBuf> {
    let safetensors_path = base_dir.join(format!("{}.safetensors", model_name));
    if safetensors_path.is_file() {
        return Some(safetensors_path);
    }

    let sub_safetensors = base_dir.join(model_name).join("model.safetensors");
    if sub_safetensors.is_file() {
        return Some(sub_safetensors);
    }

    // Alias fallback for "large"
    if model_name == "large" {
        if let Some(p) = find_model("large-v3", base_dir) {
            return Some(p);
        }
        if let Some(p) = find_model("large-v2", base_dir) {
            return Some(p);
        }
    }

    None
}


pub fn list_local_models(base_dir: &Path) -> Vec<serde_json::Value> {
    let mut list = Vec::new();
    for m in SUPPORTED_MODELS {
        let found = find_model(m.name, base_dir);
        let download_url = format!("https://huggingface.co/{}/resolve/main/{}", m.repo_id, m.file_name);
        list.push(serde_json::json!({
            "name": m.name,
            "expected_size_mb": m.size_mb,
            "download_url": download_url,
            "installed": found.is_some(),
            "path": found.map(|p| p.to_string_lossy().to_string()),
        }));
    }
    list
}
