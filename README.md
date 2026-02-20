# Classification & Segmentation of Income Survey Data

This repo implements two tasks on the U.S. Census income survey dataset:

- **Objective 1 — Income Classification:** predict whether an individual’s income is **≥ $50K**
- **Objective 2 — Segmentation:** cluster the population into **interpretable, actionable personas** for business targeting

The workflow is **reproducible**, **pipeline-driven**, and kept **minimal + functional** for the take-home:
- Core logic lives in **`src/`**
- The notebook captures EDA + modelling rationale (feature choices, metrics, plots)

---

## 📌 Highlights
- Handles **class imbalance (~6% positive)** with **PR-AUC** as the primary metric
- Uses **survey weights** (`weight`) as `sample_weight` in **training and evaluation**
- Builds personas via **One-Hot Encoding → TruncatedSVD → KMeans**
- Produces **weighted segment profiles** (size share + high-income propensity + top categories + numeric heatmap)

---

## 📁 Repository structure

```text
census-income-classification-segmentation/
├── README.md
├── data/
│   ├── processed
│   ├── raw
├── Report.pdf
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── features.py
│   ├── train_classifier.py
│   ├── eval_classifier.py
│   ├── cluster_segments.py
│   ├── profile_segments.py
│   └── utils.py
├── notebook
│   ├──notebook.ipynb
├── figs/                       
├── requirements.txt
└── .gitignore

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
Place the raw dataset file under:
data/raw/
```

Example:
```text
data/raw/census.csv
```

The pipeline expects:
	•	label → used to derive binary target (≥ $50K)
	•	weight → survey sampling weight (used for training/evaluation only, not a predictive feature)
	•	remaining columns → numeric/categorical features used by the preprocessing pipeline

Note: data/ is ignored by git (recommended for take-homes). Do not commit raw data.

## 🚀 Quickstart (run end-to-end)

Step 1 — Data prep (cleaning + feature engineering + split)

Creates:
	•	data/processed/train.csv
	•	data/processed/test.csv
	•	optional metadata files (schema / feature lists)

```bash
python -m src.data_prep \
  --raw_path data/raw/census.csv \
  --out_dir data/processed \
  --test_size 0.2 \
  --seed 42
```

Step 2 — Train classifiers (LR, RF, XGBoost)

Trains baseline models and writes the best model artifact.

```bash
python -m src.train_classifier \
  --data_dir data/processed \
  --out_dir artifacts/models \
  --seed 42
```

Step 3 — Evaluate best classifier (metrics + plots)

Writes:
	•	metrics table (ROC-AUC, PR-AUC, Precision, Recall, F1)
	•	confusion matrix (raw + weighted if enabled)
	•	ROC and PR curves

```bash
python -m src.eval_classifier \
  --data_dir data/processed \
  --model_path artifacts/models/best_model.joblib \
  --out_dir artifacts/eval
```

Step 4 — Train segmentation model (SVD + KMeans)

Creates cluster assignments and stores cluster artifacts.

```bash
python -m src.cluster_segments \
  --data_dir data/processed \
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

## 🧠 Methodology summary

Objective 1 — Income Classification

Goal: rank and classify individuals as income ≥ $50K.

Key design choices:
	•	Strong class imbalance (~6% positive): evaluation emphasizes PR-AUC, not accuracy.
	•	Survey weight is used as sample_weight in training and evaluation to better reflect population-level performance.
	•	Models compared:
	•	Logistic Regression (interpretable baseline)
	•	Random Forest (nonlinear bagging)
	•	XGBoost (boosted trees; strongest tabular baseline)
	•	Hyperparameter-tuned XGBoost (light tuning; kept only if it improves PR-AUC)

Example test-set metrics (threshold=0.5):

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9477 | 0.6301 | 0.5192 | 0.7326 | 0.4020 |
| Random Forest | 0.9497 | 0.6610 | 0.4772 | 0.8168 | 0.3370 |
| XGBoost | 0.9565 | 0.6989 | 0.6005 | 0.7587 | 0.4969 |
| Hyperparameter-tuned XGBoost | 0.9527 | 0.6746 | 0.5500 | 0.7743 | 0.4265 |

Threshold note: probability threshold is a deployment knob:
	•	higher threshold → higher precision, lower recall
	•	lower threshold → higher recall, lower precision

Objective 2 — Segmentation (Unsupervised Clustering)

Goal: create interpretable personas to support targeting and messaging strategies.

Approach:
	•	Mixed numeric + categorical representation:
	•	numeric: median imputation + scaling
	•	categorical: impute Unknown + one-hot encode
	•	High-dimensional sparse matrix → TruncatedSVD (PCA analogue for sparse data)
	•	Clustering algorithm: KMeans for scalability and interpretability
	•	k selection: elbow + internal metrics (Silhouette / CH / DB) + persona interpretability
	•	Stability: reruns across seeds and checks Adjusted Rand Index (ARI)

Stability example:
	•	ARI mean = 0.936, ARI min = 0.839 (high consistency)

Example personas (k=6):
	•	Affluent Investors
	•	Prime Full-Time Workers
	•	Steady Workers
	•	Low-Income Workers
	•	Older Non-Workers
	•	Dependents

## 📌 Deliverables
	•	report/Report.pdf — final write-up
	•	figs/ — figures used in the report
	•	src/ — production-style pipeline scripts
	•	notebooks/ — exploration (EDA / modeling / segmentation)

## 🔁 Reproducibility

All scripts accept a --seed argument and use deterministic settings where possible.
