# CLIP vs. SigLIP for Hateful-Meme Classification

This project evaluates frozen CLIP and SigLIP representations on Meta's Hateful Memes benchmark. It is designed as a controlled, reproducible comparison rather than a claim that a single model is ready for content moderation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmarquez90/clip-siglip-memetic-analysis/blob/master/hateful-memes-clip-siglip.ipynb)

The experiment evaluates three distinct model families:

- an end-to-end Concat BERT literature baseline using BERT and ResNet-152;
- six primary fusion-specific linear probes: concat, align, and cross for both CLIP and SigLIP; and
- six direct nonlinear fusion classifiers over the same frozen encoders as no-LPE ablations.

The ablations reveal whether performance actually benefits from multimodal information and whether learned feature interactions improve on raw concatenation. Hyperparameters and early-stopped checkpoints are selected on the development split, and final metrics are computed on the held-out test split using continuous probabilities.

## What the notebook demonstrates

- paired image/text ingestion and validation;
- pretrained CLIP and SigLIP inference through Hugging Face Transformers;
- normalized representation extraction and local caching;
- linear-probe prediction over concatenation, element-wise alignment, and full cross-modal interaction;
- direct nonlinear fusion classifiers as matched no-LPE ablations;
- a separately implemented Concat BERT literature baseline in models/concat_bert.py;
- reusable projected-fusion architectures and training in models/fusion.py;
- probability-based ROC-AUC and average precision;
- accuracy, balanced accuracy, precision, recall, F1, and confusion matrices;
- qualitative inspection of confidently wrong predictions;
- separate checkpoints, metrics, predictions, training histories, and environment metadata.

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
3. While the repository is private, add a fine-grained GitHub token with read-only Contents access to Colab Secrets as `GH_TOKEN`, then enable notebook access to that secret. This step is unnecessary after the repository becomes public.
4. Run the first code cell. It keeps Colab's preinstalled scientific stack, installs the model-specific dependencies, mounts Google Drive, and clones this repository only when the external model modules are not already available. The token is passed to Git through a temporary authorization header and is never saved in the notebook or clone URL.
5. Replace `<YOUR_PROJECT_FOLDER>` in `COLAB_DATA_ROOT` and `COLAB_ARTIFACT_ROOT` with a folder in your own Drive.
6. Select **Runtime > Run all**. If the dataset is absent, review the linked license and type `I ACCEPT` at the prompt. KaggleHub downloads and extracts approximately 3.6 GB into Drive as a single archive transfer. Existing complete files skip the download.

In Colab, the bootstrap installs the latest compatible Transformers 4.x and SentencePiece releases used by this notebook. It installs a core runtime package only if Colab does not already provide it, so it does not hot-swap NumPy, pandas, scikit-learn, seaborn, or PyTorch inside a running kernel. This preserves both binary compatibility and Colab's CUDA-enabled PyTorch build. In a local checkout, the bootstrap installs the complete pinned environment from `requirements.txt`.

If an earlier version of the install cell replaced NumPy or another compiled package, choose **Runtime > Disconnect and delete runtime**, reopen the notebook, and run it from the first cell. Restarting code in the already-modified runtime is not sufficient.

Drive-backed artifacts allow cached CLIP and SigLIP embeddings to survive runtime disconnects. To use ephemeral Colab storage instead, set `HATEFUL_MEMES_ARTIFACT_ROOT` to `/content/artifacts` before running the configuration cell.

Advanced users can set `SKIP_NOTEBOOK_INSTALL=true` or `SKIP_GOOGLE_DRIVE_MOUNT=true` in the environment before running the bootstrap. If a Colab session already has an authenticated checkout of the repository, set `NOTEBOOK_REPO_ROOT` to that checkout before running the bootstrap; this avoids a second clone and also works while the repository is private.

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

- `clip_fusion_lpe_metrics.csv` and `siglip_fusion_lpe_metrics.csv` — the three primary fusion-specific linear probes for each frozen encoder;
- `fusion_lpe_metrics.csv` and `fusion_lpe_test_predictions.csv` — combined results and probabilities for the six LPE systems;
- `direct_fusion_ablation_metrics.csv` and `direct_fusion_ablation_predictions.csv` — results for the six nonlinear no-LPE ablations;
- `test_predictions.csv` — combined primary, ablation, and baseline probabilities for transparent error analysis;
- `concat_bert_metrics.csv`, `concat_bert_training_history.csv`, and `concat_bert_test_predictions.csv` — the independent literature-baseline run;
- `primary_model_metrics.csv` — the ordered seven-model comparison table;
- `primary_model_bootstrap_intervals.csv` — held-out uncertainty intervals for the seven primary models;
- `run_metadata.json` — versions, device, seed, and model identifiers;
- `embeddings/` — reusable frozen embeddings; and
- `models/` — one PyTorch checkpoint per fusion-specific LPE, direct-fusion ablation, and Concat BERT baseline.

Do not publish performance claims until the notebook has completed and the generated metrics have been reviewed. Cached artifacts are intentionally excluded from version control because they can be large and may encode dataset-derived information.

## Methodological notes

- Both encoders receive the same dataset splits and fusion/readout families.
- Embeddings are L2-normalized before fusion and linear probing.
- Concat and align LPE use linear layers over fixed fused features. Cross LPE uses a bilinear layer, which is linear over the full outer-product feature map without materializing it.
- LPE checkpoints and direct nonlinear fusion checkpoints are selected independently by development ROC-AUC before held-out evaluation.
- The direct CLIP/SigLIP fusion heads use frozen cached embeddings and are reported only as no-LPE ablations.
- Concat BERT is a separate end-to-end BERT + ResNet-152 reproduction; it does not reuse CLIP/SigLIP embeddings and is not presented as an original architecture. The implementation adapts DrivenData's public mid-level concatenation scaffold, whose original demonstration used fixed fastText embeddings rather than BERT.
- ROC-AUC and average precision use probabilities, not thresholded predictions.

## Limitations and responsible use

This is a benchmark experiment, not a deployable moderation system. Real-world use would require subgroup fairness analysis, probability calibration, human review, an appeals process, adversarial evaluation, and monitoring for distribution shift. The binary label also compresses many distinct forms and targets of hate.

The source material can contain hateful and discriminatory content. Image rendering is disabled by default in the notebook and must be explicitly enabled.

## References

- Kiela et al., [The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes](https://arxiv.org/abs/2005.04790)
- DrivenData, [How to build a multimodal deep learning model to detect hateful memes](https://drivendata.co/blog/hateful-memes-benchmark)
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- Kumar and Nandakumar, [Hate-CLIPper: Multimodal Hateful Meme Classification based on Cross-modal Interaction of CLIP Features](https://arxiv.org/abs/2210.05916)
- Hee, Lee, and Chong, [On Explaining Multimodal Hateful Meme Detection Models](https://arxiv.org/abs/2204.01734)

## License

The original code in this repository is licensed under the MIT License.

The Hateful Memes dataset, pretrained model weights, and other third-party
assets remain subject to their respective licenses and are not covered by
this repository's code license.
