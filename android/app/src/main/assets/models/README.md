# Model Assets

This directory stores whisper.cpp-compatible `.bin` model files that are bundled into the APK at build time.

## How to add a new model

1. Place the `.bin` file here (e.g. `ggml-base.bin`)
2. Register it in `ModelRepository.AVAILABLE_MODELS` (`app/.../asrmobile/ModelRepository.kt`)
3. Rebuild the APK — the model will appear in the "Built-in Models" list in the app

## Current models

See `ModelRepository.kt` → `AVAILABLE_MODELS` for the current list.

## Notes

- Model files in this directory are ignored by Git by default (see `.gitignore`).
- Large models (>150MB) may cause APK size issues; prefer pushing via `adb` for experiments.
