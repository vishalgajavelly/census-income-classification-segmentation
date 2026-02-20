## Classification & Segmentation of Income Survey Data

- **Objective 1 — Income Classification:** predict whether an individual’s income is **≥ $50K**
- **Objective 2 — Segmentation:** cluster the population into **interpretable, actionable personas** for business targeting

Designed to be **reproducible**, **pipeline-driven**, and **production-style**:
- Core logic lives in **`src/`**
- Notebook includes everything from exploration to modelling 

---

## 📌 Highlights
- Handles **strong class imbalance (~6% positive)** with PR-AUC focused evaluation
- Uses **survey weights** (`weight`) as `sample_weight` in training + evaluation
- Builds **interpretable personas** via **SVD + KMeans** and validates **stability (ARI)**

---

## 📁 Repository structure

```text
census-income-classification-segmentation/
├── README.md
├── TakeHomeProject
│   ├── ML-TakehomeProject.pdf
│   ├── census_bereau_columns.csv
│   ├── census_bereau_data.csv.zip
├── Report.pdf
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── train_classifier.py
│   ├── eval_classifier.py
│   ├── cluster_segments.py
│   ├── profile_segments.py
│   └── utils.py
├── notebook.ipynb
├── figs/
│   ├── target_and_age.png
│   ├── key_distributions_log.png
│   ├── roc_pr_curves.png
│   ├── svd_cumvar.png
│   ├── cluster_metric_compare.png
│   ├── kmeans_elbow.png
│   ├── segment_numeric_heatmap.png
│   └── persona_bubble.png
├── requirements.txt
└── .gitignore
```
---

## ⚙️ Environment setup

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

## 📦 Data
```text
data/raw/
  census_bureau_data.csv
  census_bureau_columns.csv
```
The pipeline expects:
	•	label → used to derive binary target (≥ $50K)
	•	weight → survey sampling weight (used for training/eval + segment weighting; not a predictive feature)
	•	remaining columns → numeric/categorical predictors

## 🚀 Quickstart (run end-to-end)

Step 1 — Data prep (load + clean + target)

Creates:
	•	data/processed/census_clean.csv

```bash
python -m src.data_prep \
  --raw data/raw/census_bureau_data.csv \
  --columns data/raw/census_bureau_columns.csv \
  --out data/processed/census_clean.csv
```

Step 2 — Train classifiers (LR, RF, XGBoost)

Creates:
	•	artifacts/models/best_model.joblib
	•	artifacts/models/model_metrics.csv
	•	artifacts/models/train_meta.json

```bash
python -m src.train_classifier \
  --data data/processed/census_clean.csv \
  --out_dir artifacts/models \
  --seed 42
```

Step 3 — Evaluate best classifier (metrics + plots)

Creates:
	•	artifacts/eval/metrics.json
	•	artifacts/eval/metrics.csv
	•	confusion matrices + ROC/PR curves

```bash
python -m src.eval_classifier \
  --data data/processed/census_clean.csv \
  --model_path artifacts/models/best_model.joblib \
  --meta_path artifacts/models/train_meta.json \
  --out_dir artifacts/eval
```

Step 4 — Train segmentation model (SVD + KMeans)

Creates:
	•	artifacts/segments/preprocess_clust.joblib
	•	artifacts/segments/svd.joblib
	•	artifacts/segments/kmeans.joblib
	•	artifacts/segments/cluster_assignments.csv
	•	artifacts/segments/cluster_summary.csv
	•	artifacts/segments/metadata.json

```bash
python -m src.cluster_segments \
  --data data/processed/census_clean.csv \
  --out_dir artifacts/segments \
  --k 6 \
  --svd_components 50 \
  --seed 42
```

Step 5 — Profile segments (personas + visuals)

Generates segment summaries and persona plots.

```bash
python -m src.profile_segments \
  --data_dir data/processed \
  --segments_dir artifacts/segments \
  --out_dir artifacts/segments_profile
```
Step 5 — Profile segments (weighted personas + visuals)

Creates:
	•	artifacts/segments_profile/segment_profile_table.csv
	•	artifacts/segments_profile/persona_map.json
	•	artifacts/segments_profile/segment_top_categories/*.csv
	•	artifacts/segments_profile/figs/segment_numeric_heatmap.png
	•	artifacts/segments_profile/figs/persona_bubble.png

```bash
python -m src.profile_segments \
  --segments_dir artifacts/segments \
  --out_dir artifacts/segments_profile
```

## 🧠 Methodology summary

Objective 1 — Income Classification

Goal: classify individuals as income ≥ $50K.

Key choices:
	•	PR-AUC emphasized due to class imbalance
	•	Survey weights used as sample_weight for:
	•	model training (fit(..., sample_weight=weight))
	•	evaluation metrics (weighted ROC-AUC / PR-AUC / Precision / Recall / F1)
	•	Models trained:
	•	Logistic Regression (scaled numerics)
	•	Random Forest
	•	XGBoost (best tabular baseline)

Artifacts:
	•	best_model.joblib contains the full sklearn Pipeline (preprocess + estimator)
	•	model_metrics.csv compares models using the same threshold

Objective 2 — Segmentation (Unsupervised Clustering)

Goal: create interpretable personas to support targeting and messaging strategies.

Pipeline:
	•	Preprocess mixed types:
	•	numeric: median imputation + scaling
	•	categorical: impute "Unknown" + OneHotEncode
	•	Dimensionality reduction:
	•	TruncatedSVD on sparse encoded matrix (PCA analogue)
	•	Clustering:
	•	KMeans (k=6) with n_init=20 and fixed seed
	•	Profiling:
	•	weighted segment size share (weight_share)
	•	weighted income propensity (hi_rate_w)
	•	weighted numeric means (heatmap)
	•	weighted top categories for key categorical variables

Personas are assigned using lightweight heuristics to keep results report-friendly.

📌 Deliverables
	•	Report.pdf — final write-up
	•	src/ — pipeline scripts (minimal, runnable)
	•	notebook.ipynb — EDA + modeling narrative
	•	figs/ — figures used in report (optional to regenerate)


## 🔁 Reproducibility

All scripts accept a --seed argument and use deterministic settings where possible.
Training/evaluation uses the same split parameters via train_meta.json.

