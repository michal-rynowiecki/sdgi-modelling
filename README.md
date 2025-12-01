# sdgi-modelling

This repository contains the pipeline for downloading the UNDP SDGI Corpus, performing Exploratory Data Analysis (EDA), _and fine-tuning the **SmolLM** family of models for text generation and classification tasks._ (the cursive part is to be done)

## Repository Structure

```text
sdgi-modelling/
├── configs/                 # Configuration files
├── data/                    # Local dataset storage (ignored by git)
├── notebooks/               # Jupyter notebooks for EDA
├── output/                  # Model checkpoints (ignored by git)
├── scripts/                 # Execution scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Run

Follow these steps to set up the environment and execute the pipeline.

### 1. Create a Virtual Environment

**Create the environment:**
```bash
python -m venv .venv
```

**Activate the environment:**

*   **macOS, Linux:**
    ```bash
    source .venv/bin/activate
    ```

*   **Windows (Command Prompt):**
    ```cmd
    .venv\Scripts\activate.bat # Command Prompt
    .venv\Scripts\Activate.ps1 # PowerShell
    ```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download Data

Run the script to fetch the **UNDP/sdgi-corpus** from Hugging Face and save it locally to `data/raw/`.

```bash
python scripts/01_download_data.py
```
