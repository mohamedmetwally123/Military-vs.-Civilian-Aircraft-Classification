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
| 8. Train or load | loads the bundled 1× fine-tuned checkpoint with the original subset and settings | seconds |

Later runs skip the download, subset and preprocessing steps, because they are already done.

### Does it train or load the model?
Section 8 checks whether a model was **already trained on this subset with these settings**:

- **Yes, already trained** -> it **loads** the saved model from `runs/` and does **not** train again.
  Section 8 prints `Finished model found: will LOAD it (no training)`.
- **No, not trained yet** (a different subset, or different training settings) -> it **trains a new model**
  and saves it in a new folder in `runs/`. Section 8 prints `No model yet: will TRAIN`.

"Same subset and settings" means: the subset from `make_subset.py` (e.g. `subset_4450_b187ce`) plus the
settings in sections 6-8 (`IMAGE_SIZE`, `BATCH_SIZE`, `SEED`, `CHANNELS`, `DROPOUT`, `LEARNING_RATE`,
`WEIGHT_DECAY`, `EPOCHS`, `PATIENCE`, `NEITHER_LOSS_WEIGHT`).

With the original settings, the notebook loads the [1× fine-tuned checkpoint](runs/subset_4450_b187ce__unweighted_finetuned/).
It scored 89.7% on the existing test split. In section 8.1, set `BUNDLED_CHECKPOINT = "baseline"`
to load the original model (90.1% validation accuracy), or use `"accuracy_finetuned"` to load the
[classifier-only experiment selected by overall validation accuracy](runs/subset_4450_b187ce__accuracy_finetuned/). Setting `NEITHER_LOSS_WEIGHT = 2.0` creates a separate
training run that doubles the loss contribution of each bird or drone image. Section 11 reports
recall and prediction counts for birds and drones separately.

The [weighted fine-tuning run](runs/subset_4450_b187ce__weighted_head/) contains the earlier 2× experiment.
It froze the feature extractor and selected the epoch by validation F1 for `neither`.

## Demonstration and correctness checks

The core algorithm in sections 6–8 batches preprocessed RGB images, extracts features with four
convolutional blocks, and produces three class scores. During training, cross-entropy and
backpropagation update the CNN. For the demonstration, section 8 loads the saved weights, section 9
shows validation predictions, and section 11 scores and displays predictions on the held-out images.

Run sections 1–8, 9.1, and 11 with the original settings and the default 1× checkpoint to reproduce
these checks on the original dataset split:

| Check | Expected result |
|---|---|
| Three-class inference on the 445-image test split | 399 correct. The diagonal of the confusion matrix is 183 commercial, 157 military, and 59 `neither`. |
| Bird and drone label mapping and predictions | `bg_birds` and `bg_drones` map to `neither`. Section 11 reports 29 of 38 birds and 30 of 39 drones correctly labeled `neither`. |

Section 11 writes the [test report](runs/subset_4450_b187ce__unweighted_finetuned/test_results.json)
and [confusion matrix](runs/subset_4450_b187ce__unweighted_finetuned/test_confusion_matrix.png).

## Team progress

### Contributions

| Member | Repository record |
|---|---|
| Mohamed Metwally | Added dataset sampling and preprocessing, the notebook's data loading and CNN training, bundled checkpoints, and setup instructions. |
| Jarod Chan | Added the model evaluation sections to the notebook. |
| Giovanna Sanchirico | Tested and fine-tuned the classifier, verified the default checkpoint, and documented the test results and team progress. |

### What remains

- Evaluate the original and 1× fine-tuned checkpoints on a fresh labeled photo set. The team has already used the existing test split for several comparisons.
- Review military-versus-`neither` errors and collect more varied examples of birds and drones. That confusion accounted for 40 of 46 mistakes in the default checkpoint's evaluation.

### Challenges encountered

- The training set had 616 `neither` photos and 1,472 photos in each aircraft class. The original model classified only 45 of 77 bird and drone photos correctly. Fine-tuning raised that to 59 of 77, but 18 were still classified as military.
- The model with the highest validation accuracy differed from the model with the highest score on the existing test split. The original scored 90.1% on validation, and the 1× fine-tune scored 89.7% on the existing test split.
- Dataset download and CPU training took substantial time. Bundled checkpoints let the team reproduce inference and testing without repeating training.

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
