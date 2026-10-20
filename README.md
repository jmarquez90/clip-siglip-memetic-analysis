# CLIP vs. SigLIP for Hateful-Meme Classification

This project evaluates frozen CLIP and SigLIP representations on Meta's Hateful Memes benchmark. It is designed as a controlled, reproducible comparison rather than a claim that a single model is ready for content moderation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmarquez90/clip-siglip-memetic-analysis/blob/master/hateful-memes-clip-siglip.ipynb)

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

The dataset is not included in this repository. The notebook includes a license-acknowledgement cell that downloads the public [`parthplc/facebook-hateful-meme-dataset` Kaggle mirror](https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset) as one archive. This avoids the per-file request limits triggered by repositories that store 10,000 images individually. Kaggle's `test.jsonl` is the unlabeled challenge file, so the notebook reuses a previously downloaded `test_seen.jsonl` when available or downloads that one small labeled annotation file from a pinned Hugging Face revision. Before downloading, the notebook links to and requires explicit acceptance of the [Hateful Memes Dataset License Agreement](https://www.drivendata.org/competitions/64/hateful-memes/).

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

`HATEFUL_MEMES_DATA_ROOT` may point either to `<HATEFUL_MEMES_DATA_ROOT>` in the example above or directly to its `hateful-memes` child directory. The loader accepts the original `dev_seen.jsonl` and `test_seen.jsonl` names as well as the Kaggle mirror's `dev.jsonl` and `test.jsonl` names.

The benchmark deliberately uses only the original `text` and binary `label` fields. Gold protected-class and attack annotations are excluded because they may be unavailable at inference time and could act as privileged information.

## Run in Google Colab

The notebook includes a self-contained Colab bootstrap and can be opened directly from GitHub without cloning the repository first.

1. Open `hateful-memes-clip-siglip.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > GPU**.
3. Run the first code cell. It keeps Colab's preinstalled scientific stack, installs the model-specific dependencies, and mounts Google Drive.
4. Replace `<YOUR_PROJECT_FOLDER>` in `COLAB_DATA_ROOT` and `COLAB_ARTIFACT_ROOT` with a folder in your own Drive.
5. Select **Runtime > Run all**. If the dataset is absent, review the linked license and type `I ACCEPT` at the prompt. KaggleHub downloads and extracts approximately 3.6 GB into Drive as a single archive transfer. Existing complete files skip the download.

In Colab, the bootstrap installs the latest compatible Transformers 4.x and SentencePiece releases used by this notebook. It installs a core runtime package only if Colab does not already provide it, so it does not hot-swap NumPy, pandas, scikit-learn, seaborn, or PyTorch inside a running kernel. This preserves both binary compatibility and Colab's CUDA-enabled PyTorch build. In a local checkout, the bootstrap installs the complete pinned environment from `requirements.txt`.

If an earlier version of the install cell replaced NumPy or another compiled package, choose **Runtime > Disconnect and delete runtime**, reopen the notebook, and run it from the first cell. Restarting code in the already-modified runtime is not sufficient.

Drive-backed artifacts allow cached CLIP and SigLIP embeddings to survive runtime disconnects. To use ephemeral Colab storage instead, set `HATEFUL_MEMES_ARTIFACT_ROOT` to `/content/artifacts` before running the configuration cell.

Advanced users can set `SKIP_NOTEBOOK_INSTALL=true` or `SKIP_GOOGLE_DRIVE_MOUNT=true` in the environment before running the bootstrap.

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

## License

The original code in this repository is licensed under the MIT License.

The Hateful Memes dataset, pretrained model weights, and other third-party
assets remain subject to their respective licenses and are not covered by
this repository's code license.