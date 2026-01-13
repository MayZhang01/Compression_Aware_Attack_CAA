# Compression-Aware Attack for LLaLA

This repository evaluates large vision-language models (LVLMs) for robustness under visual token compression using a Compression Aware Attack (CAA).

### Purpose
- Assess how compression (token retention rates) affects LVLM performance under adversarial visual perturbations.

### Quick overview
- Main entry: `llava_eval_pope.py` — example evaluation script with CAA parameters for the POPE dataset.
- Adversarial generation: `llava/eval/find_adv_patch.py` contains the CAA implementation that generates adversarial perturbations.
- Evaluation runner: `llava/eval/model_vqa_loader_adv_i.py` calls the adversarial generator and evaluates model performance across different retention rates.
- Dependencies: listed in `requirements.txt`.
- Datasets: the `dataset/` folder contains question files (JSONL) used by the runners.

### How to run (example)
1. Install requirements:

```
pip install -r requirements.txt
```

2. Configure GPU and model path (edit `llava_eval_pope.py` ArgsDefault or pass args):

```
export CUDA_VISIBLE_DEVICES=0
python llava_eval_pope.py
```

### Notes
- Edit `ArgsDefault` in `llava_eval_pope.py` to change `model_path`, `question_file`, `image_folder`, and attack params (epsilon, steps, retention rates).
- `llava/eval/model_vqa_loader_adv_i.py` runs evaluations at different retention rates and reports attack effectiveness.
- `requirements.txt` contains the Python packages required to run the evaluation scripts.


