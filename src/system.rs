//! Hardware scanner and Whisper model recommendation engine with GPU detection.

use serde::Serialize;
use std::process::Command;
use sysinfo::System;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Serialize, Clone)]
pub struct ModelRecommendation {
    pub model: &'static str,
    pub min_ram_gb: f32,
    pub recommended_ram_gb: f32,
    pub can_run: bool,
    pub performance: &'static str,
    pub note: &'static str,
}

#[derive(Debug, Serialize, Clone)]
pub struct GpuInfo {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub vram_total_mb: u64,
    pub driver_version: String,
    pub compute_capability: String,
    pub is_recommended: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct ComputeDeviceOption {
    pub id: String,
    pub label: String,
    pub device_type: String,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct SystemDiagnostics {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_ram_gb: f32,
    pub available_ram_gb: f32,
    pub gpus: Vec<GpuInfo>,
    pub primary_gpu: Option<GpuInfo>,
    pub recommended_device_id: String,
    pub compute_devices: Vec<ComputeDeviceOption>,
    pub recommended_default_model: &'static str,
    pub model_recommendations: Vec<ModelRecommendation>,
}

#[cfg(target_os = "windows")]
fn inspect_windows_adapter(subkey: &str, idx: usize) -> Option<GpuInfo> {
    let mut cmd = Command::new("reg");
    cmd.args(["query", subkey]);
    cmd.creation_flags(CREATE_NO_WINDOW);

    let output = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);

    let mut desc = String::new();
    let mut provider = String::new();
    let mut driver_version = String::new();
    let mut vram_bytes: u64 = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("DriverDesc") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                desc = parts[2..].join(" ");
            }
        } else if trimmed.starts_with("ProviderName") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                provider = parts[2..].join(" ");
            }
        } else if trimmed.starts_with("DriverVersion") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                driver_version = parts[2..].join(" ");
            }
        } else if trimmed.starts_with("HardwareInformation.qwMemorySize") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if let Some(val_str) = parts.last() {
                let clean = val_str.trim_start_matches("0x");
                if let Ok(bytes) = u64::from_str_radix(clean, 16) {
                    vram_bytes = bytes;
                }
            }
        } else if trimmed.starts_with("HardwareInformation.MemorySize") && vram_bytes == 0 {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if let Some(val_str) = parts.last() {
                let clean = val_str.trim_start_matches("0x");
                if let Ok(bytes) = u64::from_str_radix(clean, 16) {
                    vram_bytes = bytes;
                }
            }
        }
    }

    if desc.is_empty()
        || desc.to_lowercase().contains("basic render")
        || desc.to_lowercase().contains("remote display")
    {
        return None;
    }

    let vendor = if desc.to_lowercase().contains("intel") || provider.to_lowercase().contains("intel") {
        "Intel".to_string()
    } else if desc.to_lowercase().contains("nvidia") || provider.to_lowercase().contains("nvidia") {
        "NVIDIA".to_string()
    } else if desc.to_lowercase().contains("amd")
        || desc.to_lowercase().contains("radeon")
        || provider.to_lowercase().contains("amd")
    {
        "AMD".to_string()
    } else {
        "Unknown".to_string()
    };

    let desc_lower = desc.to_lowercase();
    let is_dgpu = desc_lower.contains("arc")
        || desc_lower.contains("geforce")
        || desc_lower.contains("rtx")
        || desc_lower.contains("gtx")
        || desc_lower.contains("radeon rx")
        || (vram_bytes >= 3 * 1024 * 1024 * 1024);

    let device_type = if is_dgpu {
        "Discrete GPU".to_string()
    } else {
        "Integrated GPU".to_string()
    };

    let compute_capability = if vendor == "Intel" && desc_lower.contains("arc") {
        "Intel Xe-HPG / OpenVINO Native".to_string()
    } else if vendor == "NVIDIA" {
        "NVIDIA CUDA / Tensor Core".to_string()
    } else if vendor == "AMD" {
        "AMD ROCm / RDNA Vector Engine".to_string()
    } else {
        "DirectML / Standard GPU Acceleration".to_string()
    };

    let vram_mb = vram_bytes / (1024 * 1024);

    Some(GpuInfo {
        id: format!("gpu:{}", idx),
        name: desc,
        vendor,
        device_type,
        vram_total_mb: vram_mb,
        driver_version,
        compute_capability,
        is_recommended: is_dgpu,
    })
}

pub fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    #[cfg(target_os = "windows")]
    {
        let class_key = r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
        let mut cmd = Command::new("reg");
        cmd.args(["query", class_key, "/s", "/v", "DriverDesc"]);
        cmd.creation_flags(CREATE_NO_WINDOW);

        if let Ok(output) = cmd.output() {
            let text = String::from_utf8_lossy(&output.stdout);
            let mut subkeys = Vec::new();

            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with(r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\") {
                    let parts: Vec<&str> = trimmed.split('\\').collect();
                    if let Some(last) = parts.last() {
                        if last.len() == 4 && last.chars().all(|c| c.is_ascii_digit()) {
                            subkeys.push(trimmed.to_string());
                        }
                    }
                }
            }

            subkeys.sort();
            subkeys.dedup();

            for (idx, subkey) in subkeys.iter().enumerate() {
                if let Some(gpu) = inspect_windows_adapter(subkey, idx) {
                    gpus.push(gpu);
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("card") && !name.contains('-') {
                    let vram_path = entry.path().join("device").join("mem_info_vram_total");
                    let mut vram_mb = 0;
                    if let Ok(content) = std::fs::read_to_string(vram_path) {
                        if let Ok(bytes) = content.trim().parse::<u64>() {
                            vram_mb = bytes / (1024 * 1024);
                        }
                    }
                    gpus.push(GpuInfo {
                        id: format!("gpu:{}", gpus.len()),
                        name: format!("Linux Direct Rendering Adapter ({})", name),
                        vendor: "Generic DRM".to_string(),
                        device_type: if vram_mb > 2048 { "Discrete GPU".to_string() } else { "Integrated GPU".to_string() },
                        vram_total_mb: vram_mb,
                        driver_version: "kernel-drm".to_string(),
                        compute_capability: "Vulkan / OpenCL".to_string(),
                        is_recommended: vram_mb > 2048,
                    });
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        gpus.push(GpuInfo {
            id: "gpu:0".to_string(),
            name: "Apple Silicon Metal Acceleration".to_string(),
            vendor: "Apple".to_string(),
            device_type: "Unified Memory GPU".to_string(),
            vram_total_mb: 16384,
            driver_version: "Metal 3".to_string(),
            compute_capability: "Metal Performance Shaders".to_string(),
            is_recommended: true,
        });
    }

    gpus
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

    let gpus = detect_gpus();
    let primary_gpu = gpus
        .iter()
        .find(|g| g.is_recommended)
        .or_else(|| gpus.first())
        .cloned();

    let mut compute_devices = Vec::new();

    let auto_desc = if let Some(ref g) = primary_gpu {
        format!("Auto: Prioritize discrete GPU ({} - {} GB VRAM)", g.name, g.vram_total_mb / 1024)
    } else {
        format!("Auto: Optimized CPU inference ({} threads)", cpu_cores)
    };

    compute_devices.push(ComputeDeviceOption {
        id: "auto".to_string(),
        label: "Auto (Recommended Hardware)".to_string(),
        device_type: "auto".to_string(),
        description: auto_desc,
    });

    for g in &gpus {
        compute_devices.push(ComputeDeviceOption {
            id: g.id.clone(),
            label: format!("{} ({} GB VRAM)", g.name, g.vram_total_mb / 1024),
            device_type: g.device_type.clone(),
            description: format!("Vendor: {} • Driver: {} • {}", g.vendor, g.driver_version, g.compute_capability),
        });
    }

    compute_devices.push(ComputeDeviceOption {
        id: "cpu".to_string(),
        label: format!("CPU: {} ({} Threads)", cpu_brand, cpu_cores),
        device_type: "CPU".to_string(),
        description: "Multi-threaded Candle inference with Rayon & AVX2 vector SIMD".to_string(),
    });

    let recommended_device_id = if let Some(ref g) = primary_gpu {
        g.id.clone()
    } else {
        "cpu".to_string()
    };

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
        gpus,
        primary_gpu,
        recommended_device_id,
        compute_devices,
        recommended_default_model: recommended,
        model_recommendations: models,
    }
}
