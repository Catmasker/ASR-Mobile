# ASR Mobile

Android on-device Automatic Speech Recognition (ASR) project for the course topic **Deploying ASR Models on Mobile Devices**.

This project targets Android phones and focuses on practical mobile ASR deployment: model size, latency, memory use, offline inference, and recognition quality.

## What this project contains

- Android app scaffold in [android/](android/)
- Kotlin code for UI, audio recording, model file handling, and benchmarking
- Native/JNI placeholders for integrating `whisper.cpp`
- Documentation for Android deployment and benchmarking
- Team contribution and GitHub URL templates required by the course project

## Important: no model downloads

Model weights are **not included** and are **not downloaded automatically**. If a model download is needed, ask the project owner first.

See [MODEL_SETUP.md](MODEL_SETUP.md) for manual model placement instructions.

## Recommended ASR runtime

The primary deployment path is `whisper.cpp` because it supports quantized Whisper models and CPU-only Android inference through the Android NDK/JNI.

The existing course script [whisper_audio_to_txt.py](../machine_learning_2026_spring/session-402-audio-whisper-tts/whisper_audio_to_txt.py) is useful as a desktop ASR baseline, but it uses Python `faster_whisper` and is not directly packaged into Android.

## Quick start

1. Open [android/](android/) in Android Studio.
2. Install Android SDK/NDK/CMake if Android Studio asks for them.
3. Manually provide `whisper.cpp` source under [android/app/src/main/cpp/third_party/whisper.cpp/](android/app/src/main/cpp/third_party/whisper.cpp/). Do not download it through this project without approval.
4. Use the bundled tiny model for a simple APK demo, or manually provide another whisper.cpp-compatible model file. See [MODEL_SETUP.md](MODEL_SETUP.md).
5. Build and run the Android app.
6. Tap **Use bundled tiny model**, then **Load model**, record audio, transcribe, and run benchmarks.

Until `whisper.cpp` and a model are provided, the app scaffold is designed to report that the native backend is not configured.

## Project structure

```text
ASR Mobile/
├── android/                 # Android/Kotlin/NDK scaffold
├── docs/                    # Architecture and deployment notes
├── models/                  # Whisper.cpp GGML/GGUF model files
├── model_benchmark/         # PC benchmark suite & results
├── model_eval/              # Python evaluation scripts & results
├── samples/                 # Local sample audio placeholder; audio files ignored by Git
├── scripts/                 # Safe helper scripts; no model downloads
├── BENCHMARKING.md
├── MODEL_SETUP.md
├── PROJECT_REPORT.md
├── TEAM_CONTRIBUTIONS.md
└── GITHUB_REPOSITORY.md
```

## Evaluation goals

Benchmark at least:

- model file size
- model load time
- audio duration
- transcription time
- real-time factor
- approximate memory use
- device model / Android version / ABI
- transcript quality for English, French, and Chinese samples when available

Use [BENCHMARKING.md](BENCHMARKING.md) and [docs/BENCHMARK_RESULTS_TEMPLATE.md](docs/BENCHMARK_RESULTS_TEMPLATE.md).

## Final video outline

The course asks for an English video of at least 10 minutes. Use [docs/VIDEO_SCRIPT_OUTLINE.md](docs/VIDEO_SCRIPT_OUTLINE.md) as a suggested structure.
