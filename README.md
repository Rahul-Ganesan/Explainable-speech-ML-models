# Explainable Speech ML Models

This repository contains a full **multimodal machine learning pipeline** for predicting emotional states (`Overall` and `Excited`) from prosodic and textual features extracted from speech. The pipeline supports:

- **Prosodic feature processing** (pitch, intensity, etc.)
- **Textual feature processing** (transcripts)
- **Feature selection** using Mutual Information and Random Forest
- **Independent modeling** (Decision Tree and MLP)
- **Multimodal modeling** (Feature Fusion + Model Fusion)
- Logging and reproducible experiments

---

## Directory Structure

```
├── data/
│   ├── prosodic_features.csv
│   ├── transcripts.csv
│   ├── scores.csv
│   ├── all_q_cleaned_prosodic_features.csv
│   ├── avg_cleaned_prosodic_features.csv
│   └── text_cleaned_features.csv
├── results/          # Model outputs (ignored in git)
├── src/
│   ├── prosodic_data_cleaning.py
│   ├── textual_data_cleaning.py
│   ├── feature_selection.py
│   ├── modeling.py
│   ├── multimodal_modeling.py
│   └── explainable.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Create a virtual environment and activate it:

```bash
python -m venv ml_env
# Windows
ml_env\Scripts\activate
# macOS/Linux
source ml_env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Run the full pipeline**:

```bash
python main.py
```

This will:

- Clean prosodic and textual data
- Perform feature selection
- Train independent models on prosodic and textual features
- Run multimodal modeling (feature fusion + model fusion)
- Save cleaned data to `data/` and model results to `results/`

2. **Logging**:

By default, logs are printed to the console. To save logs to a file, modify `main.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='pipeline.log',    # Logs will be written here
    filemode='w'
)
```

---

## Customizing Paths

- Raw data files must be placed in `data/`:

```
data/prosodic_features.csv
data/transcripts.csv
data/scores.csv
```

- Cleaned datasets are saved to `data/` as:

```
all_q_cleaned_prosodic_features.csv
avg_cleaned_prosodic_features.csv
text_cleaned_features.csv
```

- Model results are saved to `results/`.

---

## Adding New Features or Models

1. Add new cleaning or transformation scripts in `src/`.
2. Update `main.py` to call your new functions.
3. Use `modeling.py` and `multimodal_modeling.py` to integrate new models.
4. Add new experiments to `run_feature_selection()` and `run_modeling()`.

---

## Notes

- `.gitignore` excludes `__pycache__/` and `results/` to keep the repository clean.
- Ensure `data/` contains all necessary CSVs before running.
- The pipeline supports **multi-output regression** for predicting `Overall` and `Excited` simultaneously.

---

## Author

- **Aidan Bagley** – Aerospace Engineering / Autonomous Systems   
- **John Maddox** – [Optional affiliation]  
- **Rahul Ganesan** – [Optional affiliation]  
- **Dheeraj Gajula** – [Optional affiliation]
