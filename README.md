# HWTCMBench

A Comprehensive Benchmark for Evaluating Large Language Models in Traditional Chinese Medicine.

# Changelog
- 2024-08-09: debut

# Dataset

The dataset is available at https://huggingface.co/datasets/Monor/hwtcm

## Benchmarking model accuracy

|   | multiple-choice questions（单选题）  | multiple-answers questions（多选题）   | True/False questions（判断题） | 
|---|---|---|---|
| llama3:8b  | 21.94%  | 17.71%  | 46.56%  |
| phi3:14b-instruct  | 26.93%  | 1.04%  | 38.93%  |
| aya:8b  | 17.85%  | 1.04%  | 34.35%  |
| mistral:7b-instruct  | 21.76%  | 2.08%  | **48.09%**  |
| qwen1.5-7b-chat  | 51.35%  | 13.54%  | 46.56%  |
| qwen1.5-14b-chat | 69.94%  | **78.12%**  | 31.30%  |
| huangdi-13b-chat | 21.73%  | 45.83%  | 0.00%  |
| canggong-14b-chat(SFT)<br>**Ours** | 55.98%  | 4.17%  | 23.66%  |
| canggong-14b-chat(DPO)<br>**Ours** | **72.33%**  | 2.08%  | 45.80%  |


> canggong-14b-chat is an LLM of traditional Chinese medicine still in training.