# Military vs. Civilian Aircraft Classification

A CNN, trained from scratch with PyTorch, that sorts aircraft photos into **3 classes**:

| Class | What it is |
|---|---|
| `military` | military aircraft (fighters, bombers, transports, helicopters, UAVs) |
| `civilian` | commercial airliners |
| `neither` | birds and drones in the sky (no aircraft) |

Data: [Ahnuf/Military_Aircraft_Detection_Classification_Image_Dataset](https://huggingface.co/datasets/Ahnuf/Military_Aircraft_Detection_Classification_Image_Dataset) on Hugging Face.
Everything runs from one notebook: **`main.ipynb`**.

## Authors
- Jarod Chan
- Mohamed Metwally
- Giovanna Sanchirico

## Setup

### What you need
- **Python 3.9 or newer**
- **About 4 GB of free disk space** (the dataset download is about 2.9 GB)
- An internet connection for the first run (to download the dataset)
- A GPU is optional. On an Apple Silicon Mac the notebook uses the Mac GPU (`mps`) automatically,
  with an NVIDIA GPU it uses `cuda`, otherwise it runs on the CPU (slower, but it works).

### 1. Get the code
```bash
git clone https://github.com/mohamedmetwally123/Military-vs.-Civilian-Aircraft-Classification.git
cd Military-vs.-Civilian-Aircraft-Classification
```

### 2. Create a virtual environment and install the packages
**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Check that it worked:
```bash
python -c "import torch, torchvision; print(torch.__version__)"
```

### 3. Open the notebook
- **VS Code:** open `main.ipynb` and choose the `.venv` kernel (top right of the notebook).
- **Jupyter:** run `jupyter notebook` and open `main.ipynb` (use the `.venv` Python).

### 4. Run it
Click **Run All**. What happens on the **first** run:

| Step | What it does | Time |
|---|---|---|
| 2. Download | downloads `dataset.zip` (~2.9 GB) into `data/raw/` and unzips it | depends on your internet |
| 3. Subset | picks the 4,450 images used for the project (same pick on every computer) | seconds |
| 4. Preprocess | crops the watermark strip and resizes the images into `data/processed/` | a few minutes |
| 6. Load data | computes the normalization numbers, builds the batches | seconds |
| 8. Train or load | **loads the trained model from `runs/`** (no training needed) | seconds |

Later runs skip the download, subset and preprocessing steps, because they are already done.

### Does it train or load the model?
Section 8 checks whether a model was **already trained on this subset with these settings**:

- **Yes, already trained** -> it **loads** the saved model from `runs/` and does **not** train again.
  Section 8 prints `Finished model found: will LOAD it (no training)`.
- **No, not trained yet** (a different subset, or different training settings) -> it **trains a new model**
  and saves it in a new folder in `runs/`. Section 8 prints `No model yet: will TRAIN`.

"Same subset and settings" means: the subset from `make_subset.py` (e.g. `subset_4450_b187ce`) plus the
settings in sections 6-8 (`IMAGE_SIZE`, `BATCH_SIZE`, `SEED`, `CHANNELS`, `DROPOUT`, `LEARNING_RATE`,
`WEIGHT_DECAY`, `EPOCHS`, `PATIENCE`). Together they decide the run folder name,
e.g. `runs/subset_4450_b187ce__7a9e53/`.

With the files as they are in this repo, the model **is** already trained (it's included in `runs/`),
so you only need to run the notebook, not train anything.

## Good to know
- **Cells marked `DEMO ONLY` or `CHECK ONLY`** only show what is happening. They are safe to skip.
- **Changing settings:** the settings sit at the top of each section (and at the top of `make_subset.py`
  and `preprocess.py`). After editing `make_subset.py` or `preprocess.py`, **restart the kernel**, then Run All,
  otherwise the notebook keeps using the old values.
- **Training a new model:** change a training setting in section 8.1 (e.g. `EPOCHS`). Each combination of
  settings gets its own folder in `runs/`, so older models are kept. If you change the **code** without
  changing a setting, set `FORCE_RETRAIN = True` once.
- **Unfinished training runs are deleted automatically**, so only finished models are ever kept.

## Troubleshooting
| Problem | Fix |
|---|---|
| `NameError: name '...' is not defined` | A cell depends on an earlier one. Click **Run All** (or run the cells from the top). |
| Section 8 says `will TRAIN` instead of `will LOAD` | Your subset or settings differ from the saved runs. Check that `make_subset.py` and the section 8.1 settings were not changed. |
| Notebook can't find `torch` | The notebook is using the wrong Python. Select the `.venv` kernel. |
| Changes to `make_subset.py` have no effect | Restart the kernel, then Run All. |

## Project files
| File / folder | What it is |
|---|---|
| `main.ipynb` | the whole project: download, preprocessing, data loading, the CNN, training |
| `make_subset.py` | picks the images used (how many, and how much of each class) |
| `preprocess.py` | crops and resizes the picked images |
| `requirements.txt` | the Python packages to install |
| `runs/` | the trained models (`best.pt`), their settings, loss history and learning-curve graphs |
| `data/` | created by the notebook (dataset, subsets, processed images). Not in git |
