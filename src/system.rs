//! Hardware scanner and Whisper model recommendation engine.

use serde::Serialize;
use sysinfo::System;

#[derive(Debug, Serialize, Clone)]
pub struct ModelRecommendation {
    pub model: &'static str,
    pub min_ram_gb: f32,
    pub recommended_ram_gb: f32,
    pub can_run: bool,
    pub performance: &'static str,
    pub note: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SystemDiagnostics {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_ram_gb: f32,
    pub available_ram_gb: f32,
    pub recommended_default_model: &'static str,
    pub model_recommendations: Vec<ModelRecommendation>,
}

pub fn scan_system() -> SystemDiagnostics {
    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
    let avail_ram = sys.available_memory() as f32 / (1024.0 * 1024.0 * 1024.0);

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let cpu_cores = sys.cpus().len();

    let models = vec![
        ModelRecommendation {
            model: "tiny",
            min_ram_gb: 1.0,
            recommended_ram_gb: 2.0,
            can_run: avail_ram >= 1.0,
            performance: "Ultra Fast (~1-2s)",
            note: "Ideal for low-end hardware, basic lyrics synchronization.",
        },
        ModelRecommendation {
            model: "base",
            min_ram_gb: 2.0,
            recommended_ram_gb: 4.0,
            can_run: avail_ram >= 2.0,
            performance: "Fast & Balanced (~2-4s)",
            note: "Default recommended model for music lyrics & video subtitles.",
        },
        ModelRecommendation {
            model: "small",
            min_ram_gb: 4.0,
            recommended_ram_gb: 8.0,
            can_run: avail_ram >= 3.5,
            performance: "High Accuracy (~4-8s)",
            note: "Excellent for multi-lingual songs, acoustic tracks, and complex speech.",
        },
        ModelRecommendation {
            model: "medium",
            min_ram_gb: 6.0,
            recommended_ram_gb: 12.0,
            can_run: avail_ram >= 5.5,
            performance: "Near-Perfect Accuracy (~8-15s)",
            note: "Best accuracy for professional subtitle sync.",
        },
        ModelRecommendation {
            model: "large-v2",
            min_ram_gb: 8.0,
            recommended_ram_gb: 16.0,
            can_run: avail_ram >= 7.5,
            performance: "Near Studio Grade (~15-20s)",
            note: "80-channel mel model with maximum accuracy for complex songs.",
        },
        ModelRecommendation {
            model: "large-v3",
            min_ram_gb: 8.0,
            recommended_ram_gb: 16.0,
            can_run: avail_ram >= 7.5,
            performance: "Highest Quality Studio Grade (~15-25s)",
            note: "Heavy model with 128 mel bins for high-end systems.",
        },
    ];

    let recommended = if avail_ram >= 7.5 && cpu_cores >= 8 {
        "small"
    } else if avail_ram >= 2.0 {
        "base"
    } else {
        "tiny"
    };

    SystemDiagnostics {
        cpu_brand,
        cpu_cores,
        total_ram_gb: (total_ram * 10.0).round() / 10.0,
        available_ram_gb: (avail_ram * 10.0).round() / 10.0,
        recommended_default_model: recommended,
        model_recommendations: models,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_system() {
        let diag = scan_system();
        assert!(!diag.cpu_brand.is_empty());
        assert!(diag.cpu_cores > 0);
        assert!(diag.total_ram_gb > 0.0);
        assert!(!diag.model_recommendations.is_empty());
    }
}

