# CLIP Usage Guide

This guide summarizes how to set up and work with the CLIP models provided in this repository. It highlights the recommended environment configuration, core APIs, and the built-in examples that you can reuse for your own projects.

## Environment Setup

- Create and activate the dedicated conda environment the team uses:
  ```bash
  conda create -n CLIP python=3.9
  conda activate CLIP
  ```
- Install PyTorch (1.7.1 or newer) and torchvision. Pick the CUDA toolkit that matches your system, or use `cpuonly` on machines without a GPU:
  ```bash
  conda install -y -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
  ```
- Install the remaining Python dependencies and this package in editable mode:
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```

These steps download the CLIP weights on first use and ensure the repository's tests and examples run inside the `CLIP` conda environment.

## Quickstart: Zero-Shot Inference

Use CLIP to score images against natural-language labels without any fine-tuning:

```python
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1)

print("Label probs:", probs)
```

The helper `clip.tokenize` converts raw strings into the token IDs the text tower expects. The model returns cosine similarities (scaled by 100) between every image-text pair; applying `softmax` produces label probabilities.

## Core APIs

- `clip.available_models()` lists every bundled checkpoint (for example, `ViT-B/32`, `RN50`).
- `clip.load(model_name, device="cuda", jit=False)` downloads (if needed) and returns `(model, preprocess)` for the requested backbone.
- `clip.tokenize(texts, context_length=77)` turns one or more text prompts into the tensor format consumed by the text encoder.
- `model.encode_image(tensor)` and `model.encode_text(tensor)` expose standalone encoders, making it easy to cache features or plug CLIP into downstream tasks.
- Calling the model directly as `model(image_tensor, text_tensor)` returns the paired logits for contrastive scoring.

## Going Further

- **Zero-shot benchmarking**: `README.md` contains a CIFAR-100 example that normalizes features and ranks label prompts. Adapt it by swapping the dataset and prompt template.
- **Feature extraction + linear probing**: See the "Linear-probe evaluation" section in `README.md` for an end-to-end workflow that trains a scikit-learn logistic regression head on frozen CLIP embeddings.
- **Interactive exploration**: Launch the notebook at `notebooks/Interacting_with_CLIP.ipynb` to inspect predictions and embeddings step by step.

## Validating the Installation

Run the repository's consistency test to confirm the Python and TorchScript variants produce matching probabilities:

```bash
conda activate CLIP
pytest tests/test_consistency.py
```

The test loads every available checkpoint, runs them on `CLIP.png`, and checks that the logits align across implementations.

## Tips

- Model downloads are cached under `~/.cache/clip`. Delete individual files there if you need to redownload checkpoints.
- For CPU-only inference, install PyTorch with `cudatoolkit=cpuonly` and keep the calls on `device="cpu"`.
- To evaluate new prompts quickly, prepare them as formatted strings (`f"a photo of a {label}"`) and reuse the same tokenized batch across images.
