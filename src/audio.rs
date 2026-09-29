//! Pure Rust multi-format audio decoder and 16,000 Hz resampler using Symphonia.

use serde::Serialize;
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Serialize, Clone)]
pub struct AudioProbeInfo {
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub channels: usize,
    pub codec: String,
    pub file_size_bytes: u64,
    pub estimated_30s_chunks: usize,
}

pub fn probe_audio<P: AsRef<Path>>(path: P) -> Result<AudioProbeInfo, String> {
    let p = path.as_ref();
    let file_size_bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let src = File::open(p).map_err(|e| format!("Failed to open media file: {}", e))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| format!("Unsupported format or corrupt stream: {}", e))?;

    let format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio tracks found".to_string())?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);
    let codec = format!("{:?}", track.codec_params.codec);

    let duration_seconds = if let (Some(n_frames), Some(tb)) = (track.codec_params.n_frames, track.codec_params.time_base) {
        let time = tb.calc_time(n_frames);
        time.seconds as f64 + time.frac
    } else {
        0.0
    };

    let estimated_30s_chunks = if duration_seconds > 0.0 {
        (duration_seconds / 30.0).ceil() as usize
    } else {
        1
    };

    Ok(AudioProbeInfo {
        duration_seconds,
        sample_rate,
        channels,
        codec,
        file_size_bytes,
        estimated_30s_chunks,
    })
}

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, String> {
    let src = File::open(&path).map_err(|e| format!("Failed to open audio file: {}", e))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| format!("Unsupported audio format or probe failed: {}", e))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio tracks found in media".to_string())?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| "Audio track has unknown sample rate".to_string())?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to initialize audio decoder: {}", e))?;

    let mut raw_mono_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => {
                eprintln!("Warning: packet decode notice: {}", e);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                append_mono_samples(&decoded, channels, &mut raw_mono_samples);
            }
            Err(Error::DecodeError(e)) => {
                eprintln!("Warning: decode frame error: {}", e);
                continue;
            }
            Err(e) => {
                eprintln!("Warning: decode error: {}", e);
                break;
            }
        }
    }

    if raw_mono_samples.is_empty() {
        return Err("Decoded audio produced zero samples".to_string());
    }

    // Resample to 16,000 Hz if necessary
    if sample_rate == WHISPER_SAMPLE_RATE {
        Ok(raw_mono_samples)
    } else {
        let filtered = if sample_rate > WHISPER_SAMPLE_RATE {
            lowpass_filter(&raw_mono_samples, sample_rate, (WHISPER_SAMPLE_RATE / 2) as f32 - 500.0)
        } else {
            raw_mono_samples
        };
        Ok(linear_resample(&filtered, sample_rate, WHISPER_SAMPLE_RATE))
    }
}

fn append_mono_samples(buf: &AudioBufferRef, expected_channels: usize, out: &mut Vec<f32>) {
    let spec_channels = buf.spec().channels.count();
    let num_channels = expected_channels.min(spec_channels).max(1);
    let norm = num_channels as f32;

    match buf {
        AudioBufferRef::F32(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    sum += b.chan(c)[i];
                }
                out.push(sum / norm);
            }
        }
        AudioBufferRef::S16(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    sum += b.chan(c)[i] as f32 / 32768.0;
                }
                out.push(sum / norm);
            }
        }
        AudioBufferRef::U8(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    sum += (b.chan(c)[i] as f32 - 128.0) / 128.0;
                }
                out.push(sum / norm);
            }
        }
        AudioBufferRef::S32(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    sum += b.chan(c)[i] as f32 / 2147483648.0;
                }
                out.push(sum / norm);
            }
        }
        AudioBufferRef::F64(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..num_channels {
                    sum += b.chan(c)[i] as f32;
                }
                out.push(sum / norm);
            }
        }
        _ => {
            eprintln!("Warning: unhandled audio buffer format, skipping chunk");
        }
    }
}

/// 2nd order Butterworth low-pass filter to prevent aliasing when downsampling
pub fn lowpass_filter(input: &[f32], sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let sr = sample_rate as f32;
    let nyquist = sr / 2.0;
    let cutoff = cutoff_hz.min(nyquist * 0.95).max(100.0);

    let w0 = 2.0 * std::f32::consts::PI * cutoff / sr;
    let alpha = (w0.sin()) / (2.0f32.sqrt());
    let cos_w0 = w0.cos();

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    let b0 = b0 / a0;
    let b1 = b1 / a0;
    let b2 = b2 / a0;
    let a1 = a1 / a0;
    let a2 = a2 / a0;

    let mut output = Vec::with_capacity(input.len());
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;

    for &x0 in input {
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        output.push(y0);
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }

    output
}

/// Linear interpolating resampler
pub fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let target_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len().saturating_sub(1));
        let frac = (src_idx - idx0 as f64) as f32;

        let sample = if idx0 < input.len() {
            input[idx0] * (1.0 - frac) + input[idx1] * frac
        } else {
            0.0
        };
        output.push(sample);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_resample_ratio() {
        let input = vec![1.0; 48000];
        let resampled = linear_resample(&input, 48000, 16000);
        assert_eq!(resampled.len(), 16000);
        for s in resampled {
            assert!((s - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_lowpass_filter_dc_pass() {
        let input = vec![0.5; 1000];
        let filtered = lowpass_filter(&input, 44100, 7500.0);
        assert_eq!(filtered.len(), input.len());
        // DC value should stabilize around 0.5
        let end_sample = filtered[filtered.len() - 1];
        assert!((end_sample - 0.5).abs() < 0.05);
    }
}

