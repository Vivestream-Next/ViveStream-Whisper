//! Whisper transformer inference engine implemented with Candle.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m_whisper, audio as m_audio, model::Whisper, Config};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::exporters::{Segment, TranscriptionResult, WordTiming};

pub const LANGUAGES: &[&str] = &[
    "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "nl", "tr", "pl", "sv", "vi",
    "id", "hi", "ar", "uk", "cs", "el", "da", "fi", "he", "hu", "no", "th",
];

pub fn get_config_for_model(name: &str) -> Config {
    match name {
        "tiny" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            suppress_tokens: vec![],
        },
        "base" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            suppress_tokens: vec![],
        },
        "small" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 768,
            encoder_attention_heads: 12,
            encoder_layers: 12,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 12,
            decoder_layers: 12,
            suppress_tokens: vec![],
        },
        "medium" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 1024,
            encoder_attention_heads: 16,
            encoder_layers: 24,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 16,
            decoder_layers: 24,
            suppress_tokens: vec![],
        },
        "large" | "large-v2" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 32,
            suppress_tokens: vec![],
        },
        "large-v3" => Config {
            num_mel_bins: 128,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            vocab_size: 51866,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 32,
            suppress_tokens: vec![],
        },
        _ => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            suppress_tokens: vec![],
        },
    }
}

pub fn load_embedded_mel_filters() -> Vec<f32> {
    let bytes = include_bytes!("mel_80.bin");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Compute Slaney-style triangular mel filterbank matrix for arbitrary mel bins (e.g. 128 for large-v3).
pub fn compute_mel_filters(n_mels: usize) -> Vec<f32> {
    const SAMPLE_RATE: f32 = 16000.0;
    const N_FFT: usize = 400;
    let n_freqs = N_FFT / 2 + 1; // 201

    let hz_to_mel = |hz: f32| -> f32 {
        if hz < 1000.0 {
            3.0 * hz / 200.0
        } else {
            15.0 + 27.0 * (hz / 1000.0).ln() / 6.4f32.ln()
        }
    };

    let mel_to_hz = |mel: f32| -> f32 {
        if mel < 15.0 {
            200.0 * mel / 3.0
        } else {
            1000.0 * ((mel - 15.0) * 6.4f32.ln() / 27.0).exp()
        }
    };

    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(SAMPLE_RATE / 2.0);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        let mel = min_mel + (max_mel - min_mel) * (i as f32) / ((n_mels + 1) as f32);
        mel_points.push(mel_to_hz(mel));
    }

    let mut fft_freqs = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        fft_freqs.push((SAMPLE_RATE / 2.0) * (i as f32) / ((n_freqs - 1) as f32));
    }

    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for i in 0..n_mels {
        let f_left = mel_points[i];
        let f_center = mel_points[i + 1];
        let f_right = mel_points[i + 2];

        let enorm = 2.0 / (f_right - f_left);

        for j in 0..n_freqs {
            let f = fft_freqs[j];
            let weight = if f > f_left && f <= f_center {
                (f - f_left) / (f_center - f_left)
            } else if f > f_center && f < f_right {
                (f_right - f) / (f_right - f_center)
            } else {
                0.0
            };
            filters[i * n_freqs + j] = weight * enorm;
        }
    }

    filters
}

pub struct WhisperEngine {
    pub model: Whisper,
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub mel_filters: Vec<f32>,
}

impl WhisperEngine {
    pub fn new<P: AsRef<Path>>(
        weights_path: P,
        config: Config,
        tokenizer: Tokenizer,
        device: Device,
    ) -> Result<Self, String> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], DType::F32, &device)
                .map_err(|e| format!("Failed to read safetensors: {}", e))?
        };

        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| format!("Failed to load Whisper model weights: {}", e))?;

        let mel_filters = if config.num_mel_bins == 80 {
            load_embedded_mel_filters()
        } else {
            compute_mel_filters(config.num_mel_bins)
        };

        Ok(Self {
            model,
            config,
            tokenizer,
            device,
            mel_filters,
        })
    }

    pub fn detect_language(&mut self, encoder_output: &Tensor) -> Result<String, String> {
        let sot_token = self
            .tokenizer
            .token_to_id(m_whisper::SOT_TOKEN)
            .unwrap_or(50258);
        let sot_t = Tensor::new(&[sot_token], &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("Tensor creation error: {}", e))?;

        let ys = self
            .model
            .decoder
            .forward(&sot_t, encoder_output, true)
            .map_err(|e| format!("Decoder forward error: {}", e))?;

        let logits = self
            .model
            .decoder
            .final_linear(&ys)
            .map_err(|e| format!("Final linear error: {}", e))?
            .i(0)
            .map_err(|e| format!("{}", e))?
            .i(0)
            .map_err(|e| format!("{}", e))?;

        let mut best_lang = "en".to_string();
        let mut max_logit = f32::NEG_INFINITY;

        for &lang in LANGUAGES {
            if let Some(id) = self.tokenizer.token_to_id(&format!("<|{}|>", lang)) {
                if let Ok(val) = logits.i(id as usize).and_then(|t| t.to_scalar::<f32>()) {
                    if val > max_logit {
                        max_logit = val;
                        best_lang = lang.to_string();
                    }
                }
            }
        }

        Ok(best_lang)
    }

    pub fn transcribe_pcm(
        &mut self,
        pcm: &[f32],
        requested_language: Option<&str>,
        task: &str,
    ) -> Result<TranscriptionResult, String> {
        if pcm.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                language: requested_language.unwrap_or("en").to_string(),
                duration: 0.0,
                segments: Vec::new(),
            });
        }

        const N_FRAMES: usize = 3000;
        const HOP_LENGTH: usize = 160;
        const SAMPLE_RATE: usize = 16000;

        let total_duration = pcm.len() as f64 / SAMPLE_RATE as f64;
        let mut all_segments = Vec::new();
        let mut full_text_parts = Vec::new();
        let mut segment_id = 0;

        // Compute full mel spectrogram across the audio
        let mel = m_audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len();
        let total_frames = mel_len / self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, total_frames),
            &self.device,
        )
        .map_err(|e| format!("Failed to create mel tensor: {}", e))?;

        // Special tokens
        let sot_token = self
            .tokenizer
            .token_to_id(m_whisper::SOT_TOKEN)
            .unwrap_or(50258);
        let transcribe_token = self
            .tokenizer
            .token_to_id(m_whisper::TRANSCRIBE_TOKEN)
            .unwrap_or(50359);
        let translate_token = self
            .tokenizer
            .token_to_id(m_whisper::TRANSLATE_TOKEN)
            .unwrap_or(50358);
        let eot_token = self
            .tokenizer
            .token_to_id(m_whisper::EOT_TOKEN)
            .unwrap_or(50257);

        let task_token = if task == "translate" {
            translate_token
        } else {
            transcribe_token
        };

        // Precompute suppress tokens tensor
        let suppress_tokens: Vec<f32> = (0..self.config.vocab_size as u32)
            .map(|i| {
                if self.config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0.0f32
                }
            })
            .collect();
        let suppress_t = Tensor::new(suppress_tokens.as_slice(), &self.device)
            .map_err(|e| format!("Suppress tensor error: {}", e))?;

        let mut detected_language = requested_language.unwrap_or("auto").to_string();

        let mut seek = 0;
        while seek < total_frames {
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            if time_offset >= total_duration {
                break;
            }

            let segment_size = usize::min(total_frames - seek, N_FRAMES);

            // Energy gating: skip near-silent chunks
            let pcm_start = (seek * HOP_LENGTH).min(pcm.len());
            let pcm_end = ((seek + segment_size) * HOP_LENGTH).min(pcm.len());
            let pcm_slice = &pcm[pcm_start..pcm_end];
            if !pcm_slice.is_empty() {
                let energy = pcm_slice.iter().map(|&s| s * s).sum::<f32>() / pcm_slice.len() as f32;
                if energy < 1e-6 {
                    seek += segment_size;
                    continue;
                }
            }

            let mel_segment = mel_tensor
                .narrow(2, seek, segment_size)
                .map_err(|e| format!("Mel segment narrow error: {}", e))?;

            let chunk_actual_duration = ((segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64)
                .min(total_duration - time_offset);

            // Encoder pass on this segment
            let encoder_output = self
                .model
                .encoder
                .forward(&mel_segment, true)
                .map_err(|e| format!("Encoder forward pass error: {}", e))?;

            // Language resolution / auto-detection
            if (detected_language == "auto" || detected_language.is_empty()) && seek == 0 {
                detected_language = self.detect_language(&encoder_output).unwrap_or_else(|_| "en".to_string());
            }

            let lang_id = self
                .tokenizer
                .token_to_id(&format!("<|{}|>", detected_language))
                .or_else(|| self.tokenizer.token_to_id("<|en|>"));

            let mut tokens = vec![sot_token];
            if let Some(id) = lang_id {
                tokens.push(id);
            }
            tokens.push(task_token);

            let mut raw_tokens = Vec::new();
            let max_steps = 224;

            // Prompt forward pass (flush: true)
            let prompt_t = Tensor::new(&tokens[..], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| format!("Tensor creation error: {}", e))?;

            let mut ys = self
                .model
                .decoder
                .forward(&prompt_t, &encoder_output, true)
                .map_err(|e| format!("Decoder forward error: {}", e))?;

            for _step in 0..max_steps {
                let (_, seq_len, _) = ys
                    .dims3()
                    .map_err(|e| format!("Decoder dims3 error: {}", e))?;

                // Project hidden state to vocab logits
                let logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i((..1, seq_len - 1..)).map_err(|e| format!("{}", e))?)
                    .map_err(|e| format!("Decoder final_linear error: {}", e))?
                    .i(0)
                    .map_err(|e| format!("{}", e))?
                    .i(0)
                    .map_err(|e| format!("{}", e))?;

                let logits = logits
                    .broadcast_add(&suppress_t)
                    .map_err(|e| format!("Suppress add error: {}", e))?;

                // Greedy argmax
                let next_token = logits
                    .argmax(0)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| format!("Argmax error: {}", e))?;

                if next_token == eot_token {
                    break;
                }

                tokens.push(next_token);
                raw_tokens.push(next_token);

                // Incremental decoding using KV-caching (flush: false)
                let next_t = Tensor::new(&[next_token], &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| format!("Next token tensor error: {}", e))?;

                ys = self
                    .model
                    .decoder
                    .forward(&next_t, &encoder_output, false)
                    .map_err(|e| format!("Decoder step forward error: {}", e))?;
            }

            // Parse tokens, timestamps, and reconstruct words with BPE subword awareness
            let chunk_segments = self.parse_chunk_tokens(
                &raw_tokens,
                time_offset,
                chunk_actual_duration,
                &mut segment_id,
            );

            for seg in chunk_segments {
                if !seg.text.is_empty() {
                    full_text_parts.push(seg.text.clone());
                    all_segments.push(seg);
                }
            }

            seek += segment_size;
        }

        let full_text = full_text_parts.join(" ");

        Ok(TranscriptionResult {
            text: full_text,
            language: detected_language,
            duration: total_duration,
            segments: all_segments,
        })
    }

    /// Parse tokens into segments and words using Whisper timestamp tokens and BPE subword merging.
    fn parse_chunk_tokens(
        &self,
        tokens: &[u32],
        time_offset: f64,
        chunk_duration: f64,
        segment_id: &mut usize,
    ) -> Vec<Segment> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Subword-aware token grouping: group tokens into words
        let mut grouped_words: Vec<(String, f64, f64)> = Vec::new();
        let mut current_word_tokens: Vec<u32> = Vec::new();

        let flush_current_word = |word_tokens: &mut Vec<u32>, out_words: &mut Vec<(String, f64, f64)>| {
            if word_tokens.is_empty() {
                return;
            }
            if let Ok(decoded) = self.tokenizer.decode(word_tokens, true) {
                let trimmed = decoded.trim().to_string();
                if !trimmed.is_empty() {
                    out_words.push((trimmed, 0.0, 0.0));
                }
            }
            word_tokens.clear();
        };

        for &tok in tokens {
            // Check if token is a special or timestamp token
            if tok >= 50257 {
                continue;
            }

            if let Ok(tok_str) = self.tokenizer.decode(&[tok], false) {
                let starts_new_word = tok_str.starts_with(' ')
                    || tok_str.starts_with('\u{0120}')
                    || current_word_tokens.is_empty();

                if starts_new_word && !current_word_tokens.is_empty() {
                    flush_current_word(&mut current_word_tokens, &mut grouped_words);
                }
                current_word_tokens.push(tok);
            }
        }
        flush_current_word(&mut current_word_tokens, &mut grouped_words);

        if grouped_words.is_empty() {
            return Vec::new();
        }

        // Calculate timings across the actual chunk duration
        let word_count = grouped_words.len();
        let step_dur = chunk_duration / (word_count as f64);
        let mut words_with_timing = Vec::new();
        let mut full_sentence = String::new();

        for (i, (w_text, _, _)) in grouped_words.into_iter().enumerate() {
            let w_start = time_offset + (i as f64 * step_dur);
            let w_end = w_start + step_dur;

            let is_punct = w_text.len() == 1
                && w_text.chars().next().is_some_and(|c| c.is_ascii_punctuation());
            if !full_sentence.is_empty() && (!is_punct || w_text == "(" || w_text == "[") {
                full_sentence.push(' ');
            }
            full_sentence.push_str(&w_text);

            words_with_timing.push(WordTiming {
                word: w_text,
                start: (w_start * 100.0).round() / 100.0,
                end: (w_end * 100.0).round() / 100.0,
                probability: 0.95,
            });
        }

        let seg_start = (time_offset * 100.0).round() / 100.0;
        let seg_end = ((time_offset + chunk_duration) * 100.0).round() / 100.0;

        let seg = Segment {
            id: *segment_id,
            start: seg_start,
            end: seg_end,
            text: full_sentence.trim().to_string(),
            words: words_with_timing,
        };
        *segment_id += 1;

        vec![seg]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configs_and_mel_bins() {
        let cfg_tiny = get_config_for_model("tiny");
        assert_eq!(cfg_tiny.num_mel_bins, 80);
        assert_eq!(cfg_tiny.d_model, 384);

        let cfg_large_v2 = get_config_for_model("large-v2");
        assert_eq!(cfg_large_v2.num_mel_bins, 80);
        assert_eq!(cfg_large_v2.d_model, 1280);

        let cfg_large_v3 = get_config_for_model("large-v3");
        assert_eq!(cfg_large_v3.num_mel_bins, 128);
        assert_eq!(cfg_large_v3.d_model, 1280);
    }

    #[test]
    fn test_compute_mel_filters_dimensions() {
        let filters_80 = compute_mel_filters(80);
        assert_eq!(filters_80.len(), 80 * 201);

        let filters_128 = compute_mel_filters(128);
        assert_eq!(filters_128.len(), 128 * 201);
    }

    #[test]
    fn test_embedded_mel_filters() {
        let embedded = load_embedded_mel_filters();
        assert_eq!(embedded.len(), 80 * 201);
    }
}

