# ASR Mobile

Android on-device Automatic Speech Recognition (ASR) project for the course topic **Deploying ASR Models on Mobile Devices**.

## Models

`models/` 目录包含以下模型文件：

| Model | Size | Language | Notes |
|:-----|:----:|:--------:|:------|
| `ggml-tiny.bin` | 74 MB | Multi-lang (99 langs) | Best balance of speed & accuracy |
| `ggml-tiny.en.bin` | 74 MB | English only | Fastest English inference |
| `ggml-base-q8_0.bin` | 78 MB | Multi-lang (99 langs) | Highest accuracy under 100MB |
| `whisper-tiny-asr-mobile-fp16.gguf` | 70 MB | Multi-lang | GGUF format |

## model_benchmark

PC 端模型基准测试（whisper-cli + Python），测试模型的准确率、推理速度、RTF、内存开销。

```bash
cd model_benchmark
conda activate MachineLearning
python benchmark_models.py
```

输出：文本报告 + CSV + 可视化图表（准确率对比、速度对比、资源开销、语言准确率热图、综合排名）。

## model_eval

队友的 Python 效果评估脚本和结果（含准确率图、性能图、评估报告 JSON）。

```
model_eval/
├── eval_whisper.py        # 评估脚本
├── accuracy_*.png         # 准确率图表
├── performance_*.png      # 性能图表
├── report.json            # 评估报告
└── requirements.txt
```
