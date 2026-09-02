use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use base64::Engine;
use clap::Parser;
use colored::Colorize;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(name = "voice-prompt", about = "Neuralnet Voice Transcription & Prompt Compiler")]
struct Args {
    /// Refine spoken input into structured prompt architecture
    #[arg(short, long)]
    refine: bool,

    /// Copy resulting transcript/prompt directly to Windows clipboard
    #[arg(short, long)]
    copy: bool,

    /// Recording duration in seconds (if not set, records until Enter is pressed)
    #[arg(short, long)]
    duration: Option<u64>,

    /// DirectShow / WASAPI audio device name substring override
    #[arg(long)]
    device: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Lexicon {
    custom_vocabulary: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RefinementResult {
    pub text: String,
    pub model_resolved: String,
    pub attempts: usize,
    pub retry_history: Vec<(String, u16)>,
}

#[derive(Serialize)]
struct TelemetryLatencies {
    record: u128,
    transcribe: u128,
    refine: u128,
    total: u128,
}

#[derive(Serialize)]
struct TelemetryModels {
    transcribe: &'static str,
    refine_resolved: Option<String>,
    refine_attempts: usize,
    retries: Vec<(String, u16)>,
}

#[derive(Serialize)]
struct TelemetryTranscripts {
    raw: String,
    refined: Option<String>,
}

#[derive(Serialize)]
struct TelemetryRecord {
    id: String,
    timestamp: String,
    audio_duration_secs: f64,
    latencies_ms: TelemetryLatencies,
    models: TelemetryModels,
    transcripts: TelemetryTranscripts,
}

fn resample_mono_to_16k(input: &[i16], source_rate: u32) -> Vec<i16> {
    if source_rate == 16000 {
        return input.to_vec();
    }
    if input.is_empty() {
        return Vec::new();
    }
    let target_rate = 16000.0_f64;
    let ratio = source_rate as f64 / target_rate;
    let target_len = ((input.len() as f64) / ratio).round() as usize;
    let mut output = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let src_pos = i as f64 * ratio;
        let idx0 = src_pos.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len().saturating_sub(1));
        let frac = src_pos - idx0 as f64;
        let s0 = input[idx0] as f64;
        let s1 = input[idx1] as f64;
        let sample = (1.0 - frac) * s0 + frac * s1;
        output.push(sample.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
    }
    output
}

fn extract_transcript(val: &serde_json::Value) -> String {
    if let Some(t) = val.get("output_text").and_then(|v| v.as_str()) {
        if !t.trim().is_empty() {
            return t.trim().to_string();
        }
    }
    if let Some(t) = val.get("text").and_then(|v| v.as_str()) {
        if !t.trim().is_empty() {
            return t.trim().to_string();
        }
    }
    let mut collected = Vec::new();
    if let Some(steps) = val.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            if let Some(content) = step.get("content").and_then(|v| v.as_array()) {
                for item in content {
                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                        if !text.trim().is_empty() {
                            collected.push(text.trim());
                        }
                    }
                }
            }
            if let Some(output) = step.get("output").and_then(|v| v.as_str()) {
                if !output.trim().is_empty() {
                    collected.push(output.trim());
                }
            }
        }
    }
    if let Some(outputs) = val.get("outputs").and_then(|v| v.as_array()) {
        for item in outputs {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                if !text.trim().is_empty() {
                    collected.push(text.trim());
                }
            }
        }
    }
    collected.join(" ")
}

async fn refine_transcript_with_retry(
    client: &reqwest::Client,
    api_key: &str,
    transcript: &str,
    is_tty: bool,
) -> Result<RefinementResult, String> {
    let primary_model = "gemini-3.5-flash-lite";
    let fallback_model = "gemini-3.5-flash";

    let payload = json!({
        "system_instruction": {
            "parts": [
                {
                    "text": "You are the Neuralnet Prompt Compiler. Convert the raw spoken transcript into a concise, high-density technical prompt or instruction block. Eliminate disfluencies, false starts, and filler words. Preserve all technical tokens verbatim."
                }
            ]
        },
        "contents": [
            {
                "parts": [
                    { "text": transcript }
                ]
            }
        ]
    });

    let mut current_model = primary_model;
    let mut attempt = 0;
    let max_primary_attempts = 3;
    let mut retry_history: Vec<(String, u16)> = Vec::new();
    let start_instant = Instant::now();

    loop {
        attempt += 1;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            current_model, api_key
        );

        let res = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Refinement network error: {e}"))?;

        let status = res.status();
        let body_text = res
            .text()
            .await
            .map_err(|e| format!("Failed to read refinement response: {e}"))?;

        if status.is_success() {
            let json_val: serde_json::Value = serde_json::from_str(&body_text)
                .map_err(|e| format!("Failed to parse refinement JSON: {e}"))?;

            if let Some(text) = json_val
                .pointer("/candidates/0/content/parts/0/text")
                .and_then(|v| v.as_str())
            {
                let latency_ms = start_instant.elapsed().as_millis();

                if is_tty {
                    if current_model == primary_model && retry_history.is_empty() {
                        eprintln!(
                            "{}",
                            format!("[*] Refinement Model: {} (latency: {}ms)", current_model, latency_ms).white()
                        );
                    } else if current_model == primary_model {
                        eprintln!(
                            "{}",
                            format!(
                                "[*] Refinement Model: {} [attempt {}/{} after 503 retry] (latency: {}ms)",
                                current_model, attempt, max_primary_attempts, latency_ms
                            ).yellow()
                        );
                    } else {
                        eprintln!(
                            "{}",
                            format!(
                                "[*] Refinement Model: {} [fallback after 3x 503] (latency: {}ms)",
                                current_model, latency_ms
                            ).yellow()
                        );
                    }
                }

                return Ok(RefinementResult {
                    text: text.trim().to_string(),
                    model_resolved: current_model.to_string(),
                    attempts: attempt,
                    retry_history,
                });
            } else {
                return Err(format!("Unexpected refinement response schema: {body_text}"));
            }
        }

        let status_code = status.as_u16();
        retry_history.push((current_model.to_string(), status_code));

        if status_code == 503 {
            if current_model == primary_model && attempt < max_primary_attempts {
                let backoff_secs = 1u64 << (attempt - 1);
                if is_tty {
                    eprintln!(
                        "{}",
                        format!(
                            "[*] Primary refinement model 503 unavailable. Retrying in {}s (attempt {}/{})...",
                            backoff_secs, attempt, max_primary_attempts
                        )
                        .yellow()
                    );
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
                continue;
            } else if current_model == primary_model {
                if is_tty {
                    eprintln!(
                        "{}",
                        "[*] Primary model 503 persisted. Dynamically falling back to gemini-3.5-flash..."
                            .yellow()
                    );
                }
                current_model = fallback_model;
                attempt = 0;
                continue;
            }
        }

        return Err(format!(
            "Refinement API error ({}) with model {}: {}",
            status, current_model, body_text
        ));
    }
}

fn resolve_gemini_api_key() -> Option<String> {
    if let Ok(k) = std::env::var("GEMINI_API_KEY") {
        if !k.trim().is_empty() {
            return Some(k.trim().to_string());
        }
    }
    // Fallback: Check APPDATA/nushell/secrets.nu
    if let Ok(appdata) = std::env::var("APPDATA") {
        let secrets_path = std::path::Path::new(&appdata).join("nushell").join("secrets.nu");
        if let Ok(content) = std::fs::read_to_string(&secrets_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("$env.GEMINI_API_KEY") {
                    if let Some(idx) = trimmed.find('=') {
                        let val_part = trimmed[idx + 1..].trim();
                        let key = val_part.trim_matches(|c| c == '"' || c == '\'' || c == ';' || c == ' ');
                        if !key.is_empty() {
                            return Some(key.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

fn get_history_log_path() -> std::path::PathBuf {
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        std::path::PathBuf::from(local_app_data)
            .join("neuralnet")
            .join("voice")
            .join("history.jsonl")
    } else {
        std::path::PathBuf::from(r"C:\Users\Israel\AppData\Local\neuralnet\voice\history.jsonl")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let args = Args::parse();
    let is_tty = std::io::stdout().is_terminal();

    // 1. Validate API Key
    let api_key = match resolve_gemini_api_key() {
        Some(k) => k,
        None => {
            eprintln!(
                "{}",
                "[-] Error: GEMINI_API_KEY environment variable not set.".bold().red()
            );
            eprintln!(
                "{}",
                "[*] Please verify secrets.nu or ensure GEMINI_API_KEY is active in your environment."
                    .white()
            );
            std::process::exit(1);
        }
    };

    // 2. Load Lexicon
    let lexicon_data = include_str!("../lexicon.json");
    let lexicon: Lexicon = serde_json::from_str(lexicon_data)
        .map_err(|e| format!("Failed to parse embedded lexicon.json: {e}"))?;

    // 3. Enumerate WASAPI Audio Device
    let host = cpal::default_host();
    let input_devices = host
        .input_devices()
        .map_err(|e| format!("Failed to query input devices: {e}"))?;

    let device_list: Vec<_> = input_devices.collect();
    let selected_device = if let Some(ref override_name) = args.device {
        let needle = override_name.to_lowercase();
        device_list
            .into_iter()
            .find(|d| d.name().unwrap_or_default().to_lowercase().contains(&needle))
            .ok_or_else(|| format!("Specified audio device substring '{override_name}' not found."))?
    } else {
        // Preferred default: Webcam C920
        let c920 = device_list
            .iter()
            .find(|d| {
                let name = d.name().unwrap_or_default();
                name.contains("Microphone (HD Pro Webcam C920)")
                    || name.contains("HD Pro Webcam C920")
                    || name.contains("C920")
            })
            .cloned();

        match c920 {
            Some(d) => d,
            None => host
                .default_input_device()
                .ok_or_else(|| "No default WASAPI audio input device available.".to_string())?,
        }
    };

    let dev_name = selected_device.name().unwrap_or_else(|_| "Unknown Device".to_string());
    if is_tty {
        eprintln!(
            "{} {}",
            "[*] Audio Substrate:".bold().white(),
            dev_name.cyan()
        );
    }

    let default_config = selected_device
        .default_input_config()
        .map_err(|e| format!("Failed to inspect device default config: {e}"))?;

    let native_sample_rate = default_config.sample_rate().0;
    let native_channels = default_config.channels();
    let native_format = default_config.sample_format();

    let audio_buffer = Arc::new(Mutex::new(Vec::<i16>::new()));
    let stream_config: cpal::StreamConfig = default_config.config();

    let stream = match native_format {
        cpal::SampleFormat::F32 => {
            let buffer = Arc::clone(&audio_buffer);
            let channels = native_channels as usize;
            selected_device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = match buffer.lock() {
                        Ok(b) => b,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    for frame in data.chunks_exact(channels) {
                        let sum: f32 = frame.iter().sum();
                        let avg = sum / (channels as f32);
                        let sample_i16 = (avg.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                        buf.push(sample_i16);
                    }
                },
                |err| eprintln!("[-] WASAPI stream error: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let buffer = Arc::clone(&audio_buffer);
            let channels = native_channels as usize;
            selected_device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut buf = match buffer.lock() {
                        Ok(b) => b,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    for frame in data.chunks_exact(channels) {
                        let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                        let avg = (sum / (channels as i32)) as i16;
                        buf.push(avg);
                    }
                },
                |err| eprintln!("[-] WASAPI stream error: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let buffer = Arc::clone(&audio_buffer);
            let channels = native_channels as usize;
            selected_device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut buf = match buffer.lock() {
                        Ok(b) => b,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    for frame in data.chunks_exact(channels) {
                        let sum: f32 = frame.iter().map(|&s| s as f32 - 32768.0).sum();
                        let avg = sum / (channels as f32);
                        let sample_i16 = avg.clamp(-32768.0, 32767.0).round() as i16;
                        buf.push(sample_i16);
                    }
                },
                |err| eprintln!("[-] WASAPI stream error: {err}"),
                None,
            )?
        }
        _ => return Err("Unsupported audio sample format.".into()),
    };

    let record_start = Instant::now();
    stream
        .play()
        .map_err(|e| format!("Failed to initiate audio stream: {e}"))?;

    // 4. Capture Loop Termination
    if let Some(sec) = args.duration {
        if is_tty {
            eprintln!(
                "{}",
                format!("[*] Capturing audio for {} second(s)...", sec).white()
            );
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(sec)).await;
    } else {
        if is_tty {
            eprintln!(
                "{}",
                "[*] Recording active. Press [Enter] to stop...".bold().white()
            );
        }
        tokio::task::spawn_blocking(|| {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
        })
        .await
        .map_err(|e| format!("Stdin listener failure: {e}"))?;
    }

    drop(stream);
    let t_record = record_start.elapsed();

    let raw_samples = {
        let mut buf = audio_buffer.lock().unwrap();
        std::mem::take(&mut *buf)
    };

    if raw_samples.is_empty() {
        eprintln!("{}", "[-] Warning: No audio captured.".yellow());
        return Ok(());
    }

    // 5. In-Memory Resampling to 16,000 Hz Mono
    let final_samples = resample_mono_to_16k(&raw_samples, native_sample_rate);

    // 6. In-Memory WAV Encoding (Zero Disk I/O)
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = std::io::Cursor::new(Vec::with_capacity(final_samples.len() * 2 + 44));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| format!("Failed to initialize in-memory WavWriter: {e}"))?;
        for &sample in &final_samples {
            writer
                .write_sample(sample)
                .map_err(|e| format!("Failed writing sample: {e}"))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("Failed finalizing WAV buffer: {e}"))?;
    }

    let wav_bytes = cursor.into_inner();
    let b64_audio = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

    if is_tty {
        eprintln!(
            "{}",
            "[*] Submitting audio to gemini-3.5-transcribe via Interactions API...".white()
        );
    }

    // 7. Interactions API Request with Custom Vocabulary
    let client = reqwest::Client::builder().build()?;
    let payload = json!({
        "model": "gemini-3.5-transcribe",
        "input": [
            {
                "type": "audio",
                "data": b64_audio,
                "mime_type": "audio/wav"
            }
        ],
        "generation_config": {
            "transcription_config": {
                "custom_vocabulary": lexicon.custom_vocabulary
            }
        }
    });

    let transcribe_start = Instant::now();
    let res = client
        .post("https://generativelanguage.googleapis.com/v1beta/interactions")
        .header("x-goog-api-key", &api_key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("Interactions API network error: {e}"))?;

    let status = res.status();
    let body_text = res
        .text()
        .await
        .map_err(|e| format!("Failed reading response: {e}"))?;

    if !status.is_success() {
        return Err(format!("Transcribe API error ({status}): {body_text}").into());
    }

    let response_json: serde_json::Value = serde_json::from_str(&body_text)
        .map_err(|e| format!("Failed parsing response JSON: {e}"))?;

    if std::env::var("VOICE_PROMPT_DEBUG").is_ok() {
        eprintln!("[DEBUG API RESPONSE]: {}", serde_json::to_string_pretty(&response_json).unwrap_or_default());
    }

    let transcript = extract_transcript(&response_json);
    let t_transcribe = transcribe_start.elapsed();

    // 8. Refinement Pass (Optional)
    let (refined_result, t_refine) = if args.refine {
        if is_tty {
            eprintln!(
                "{}",
                "[*] Compiling prompt architecture with Neuralnet Refinement Engine...".white()
            );
        }
        let refine_start = Instant::now();
        let res = refine_transcript_with_retry(&client, &api_key, &transcript, is_tty).await?;
        let elapsed = refine_start.elapsed();
        (Some(res), elapsed)
    } else {
        (None, std::time::Duration::from_millis(0))
    };

    let target_copy_text = match &refined_result {
        Some(ref r) => r.text.clone(),
        None => transcript.clone(),
    };

    // 9. Non-Blocking Clipboard Injection
    if args.copy {
        if target_copy_text.trim().is_empty() {
            if is_tty {
                eprintln!("{}", "[*] Clipboard write skipped: transcript is empty.".yellow());
            }
        } else {
            let to_copy = target_copy_text.clone();
            let copy_result = tokio::task::spawn_blocking(move || {
                let mut clipboard = arboard::Clipboard::new()?;
                clipboard.set_text(to_copy)?;
                Ok::<(), arboard::Error>(())
            })
            .await;

            match copy_result {
                Ok(Ok(())) => {
                    if is_tty {
                        eprintln!("{}", "[*] Copied to clipboard.".bold().green());
                    }
                }
                Ok(Err(e)) => eprintln!("[-] Clipboard injection error: {e}"),
                Err(e) => eprintln!("[-] Clipboard task error: {e}"),
            }
        }
    }

    let t_total = total_start.elapsed();

    // 10. Atomic JSONL Telemetry Logging
    let log_path = get_history_log_path();
    let telemetry_record = TelemetryRecord {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        audio_duration_secs: t_record.as_secs_f64(),
        latencies_ms: TelemetryLatencies {
            record: t_record.as_millis(),
            transcribe: t_transcribe.as_millis(),
            refine: t_refine.as_millis(),
            total: t_total.as_millis(),
        },
        models: TelemetryModels {
            transcribe: "gemini-3.5-transcribe",
            refine_resolved: refined_result.as_ref().map(|r| r.model_resolved.clone()),
            refine_attempts: refined_result.as_ref().map(|r| r.attempts).unwrap_or(0),
            retries: refined_result.as_ref().map(|r| r.retry_history.clone()).unwrap_or_default(),
        },
        transcripts: TelemetryTranscripts {
            raw: transcript.clone(),
            refined: refined_result.as_ref().map(|r| r.text.clone()),
        },
    };

    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json_line) = serde_json::to_string(&telemetry_record) {
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
            let _ = writeln!(file, "{}", json_line);
        }
    }

    if is_tty {
        eprintln!(
            "{} {}",
            "[*] Telemetry Logged:".bold().white(),
            log_path.display().to_string().cyan()
        );
        eprintln!(
            "{}",
            format!(
                "[*] Durations: Record {:.1}s | Transcribe {:.1}s | Refine {:.1}s | Total {:.1}s",
                t_record.as_secs_f64(),
                t_transcribe.as_secs_f64(),
                t_refine.as_secs_f64(),
                t_total.as_secs_f64()
            )
            .white()
        );
    }

    // 11. Output Geometry
    if is_tty {
        println!();
        println!(
            "{} {}",
            "[Raw Transcript]:".bold().cyan(),
            transcript.cyan()
        );
        if let Some(ref refined) = refined_result {
            println!();
            println!(
                "{} \n{}",
                "[Refined Architecture]:".bold().white(),
                refined.text.green()
            );
        }
    } else {
        // Redirection / Piped Mode: Pure unadorned output
        print!("{}", target_copy_text);
    }

    Ok(())
}
