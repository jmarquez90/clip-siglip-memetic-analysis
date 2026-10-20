# CLIP vs. SigLIP for Hateful-Meme Classification

This project evaluates frozen CLIP and SigLIP representations on Meta's Hateful Memes benchmark. It is designed as a controlled, reproducible comparison rather than a claim that a single model is ready for content moderation.

The experiment trains a linear probe over three feature sets for each encoder:

- image embeddings;
- text embeddings; and
- concatenated image + text embeddings.

This ablation reveals whether performance actually benefits from multimodal information. Hyperparameters are selected on the development split, and final metrics are computed once on the held-out test split using continuous probabilities.

## What the notebook demonstrates

- paired image/text ingestion and validation;
- pretrained CLIP and SigLIP inference through Hugging Face Transformers;
- normalized representation extraction and local caching;
- controlled image-only, text-only, and multimodal ablations;
- probability-based ROC-AUC and average precision;
- accuracy, balanced accuracy, precision, recall, F1, and confusion matrices;
- qualitative inspection of confidently wrong predictions;
- saved metrics, predictions, model probes, and environment metadata.

## Dataset

The dataset is not included in this repository. Request or download it through the [Hateful Memes Challenge site](https://hatefulmemeschallenge.com/#download) and comply with its license and access terms.

Expected layout:

```text
<HATEFUL_MEMES_DATA_ROOT>/
└── hateful-memes/
    ├── train.jsonl
    ├── dev_seen.jsonl
    ├── test_seen.jsonl
    └── img/
        └── ...
```

The benchmark deliberately uses only the original `text` and binary `label` fields. Gold protected-class and attack annotations are excluded because they may be unavailable at inference time and could act as privileged information.

## Setup

Create an environment with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For a CUDA environment, install the appropriate PyTorch build for the machine before installing the remaining requirements.

Set the dataset root without putting a private path into the notebook:

```powershell
$env:HATEFUL_MEMES_DATA_ROOT = "C:\path\to\your\data"
```

Then open `hateful-memes-clip-siglip.ipynb` and run the cells from top to bottom. A GPU is recommended for embedding extraction. Cached embeddings make later probe experiments much faster.

## Outputs

Successful execution creates an ignored `artifacts/` directory containing:

- `metrics.csv` — held-out metrics for every model and feature set;
- `test_predictions.csv` — probabilities for transparent error analysis;
- `run_metadata.json` — versions, device, seed, and model identifiers;
- `embeddings/` — reusable frozen embeddings; and
- `models/` — fitted scikit-learn probes.

Do not publish performance claims until the notebook has completed and the generated metrics have been reviewed. Cached artifacts are intentionally excluded from version control because they can be large and may encode dataset-derived information.

## Methodological notes

- Both encoders receive the same dataset splits and downstream classifier family.
- Embeddings are L2-normalized before linear probing.
- Logistic-regression regularization is selected by development ROC-AUC.
- The selected probe is refit on train + development data before test evaluation.
- ROC-AUC and average precision use probabilities, not thresholded predictions.
- The constant-probability baseline makes chance-level performance explicit.

## Limitations and responsible use

This is a benchmark experiment, not a deployable moderation system. Real-world use would require subgroup fairness analysis, probability calibration, human review, an appeals process, adversarial evaluation, and monitoring for distribution shift. The binary label also compresses many distinct forms and targets of hate.

The source material can contain hateful and discriminatory content. Image rendering is disabled by default in the notebook and must be explicitly enabled.

## References

- Kiela et al., [The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes](https://arxiv.org/abs/2005.04790)
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- Kumar and Nandakumar, [Hate-CLIPper: Multimodal Hateful Meme Classification based on Cross-modal Interaction of CLIP Features](https://arxiv.org/abs/2210.05916)
- Hee, Lee, and Chong, [On Explaining Multimodal Hateful Meme Detection Models](https://arxiv.org/abs/2204.01734)

