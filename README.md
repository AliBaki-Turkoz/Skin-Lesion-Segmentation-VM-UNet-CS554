# Skin Lesion Segmentation with VM-UNet (Vision Mamba): Multi-Head SS2D and a Combined Loss

| | |
|---|---|
| **Course** | CS554 – Introduction to Machine Learning and Artificial Neural Networks, Özyeğin University (M.Sc. Computer Engineering), Spring 2025 · Instructor: Prof. Ethem Alpaydın |
| **Type** | Team project · repository maintained by Ali Baki Türköz |
| **Base code** | [VM-UNet](https://arxiv.org/abs/2402.02491) (Ruan & Xiang, 2024), official implementation, Apache-2.0. See `LICENSE` and `VM-UNet_README.md`. |
| **Task** | Binary segmentation of skin lesions on dermoscopy images (ISIC 2016 / 2017 / 2018) |

> The proposal was titled "skin lesion *classification*". The work actually carried out is lesion **segmentation**, which is what the VM-UNet code does.

## What was changed relative to VM-UNet
| Change | Where | Description |
|---|---|---|
| **Multi-head SS2D block** | `models/vmunet/vmamba.py` → `MultiHeadMamba`, used inside `VSSBlock` | 1. LayerNorm, then the input is split into *h* heads by linear projections, and each head goes through its own SS2D (selective-scan) module. 2. The heads are concatenated, then a linear fusion and LayerNorm are applied. 3. The result is added through a **gated residual** with a learnable γ (initialised to 0). |
| **Combined loss** | `utils.py` → `AdvancedLoss`, enabled in `configs/config_setting.py` | It replaces the original `BceDiceLoss`. It is 0.25·BCE-with-logits + 0.25·Tversky + 0.40·Focal-Tversky + 0.10·Log-Cosh-Dice (α = β = 0.5, γ = 1.33). |
| **ISIC 2016 support** | `configs/config_setting.py` | `datasets = 'isic16' / 'isic17' / 'isic18'` |
| **Hyperparameter search** | `hyperparameter_tuning.py` | Optuna over learning rate, weight decay, drop-path rate and early-stopping patience. The objective is the best validation loss. Results are in `optuna_study_report.csv`. |
| **Split-based training script** | `train_new.py`, `datasets/dataset_new.py` | Train/val = 80/20 of the training folder (seeded); test = the provided validation folder. Logs to Weights & Biases. |

## Data
- **Source:** ISIC 2016, 2017 and 2018 lesion segmentation data. Images with binary masks, resized to 256×256.

| | Train | Val/Test |
|---|---|---|
| ISIC 2016 | 1,052 | 270 |
| ISIC 2017 | 1,500 | 650 |
| ISIC 2018 | 1,886 | 808 |

- **Not included** (about 0.9 GB). The ISIC data have their own licence terms, see the ISIC Archive. Expected layout:
  ```
  data/isic2016/{train,val}/{images,masks}/*.png     (same for isic2017, isic2018)
  ```
- **Pretrained weights:** VMamba-Tiny ImageNet weights go in `pre_trained_weights/`. See `pre_trained_weights/README.md` for the download links from the VM-UNet authors.

## Results
**Best test results.** Source: the course Process Report (7 May 2025), Table I, and the test-output screenshots of the same runs in [`results/test_screenshots/`](results/test_screenshots). Settings: batch size 12, 130 epochs, learning rate 9e-5. Test set = the provided validation folder of each dataset.

| Dataset | Test images | Dice (F1) | mIoU | Accuracy | Specificity | Sensitivity | Screenshot |
|---|---|---|---|---|---|---|---|
| **ISIC 2016** | 270 | **0.931** | **0.870** | 0.965 | 0.980 | 0.921 | [2016](results/test_screenshots/TestResult_2016Dataset_BatchSize_12.PNG) |
| ISIC 2017 | 650 | 0.873 | 0.775 | 0.959 | 0.982 | 0.844 | [2017](results/test_screenshots/TestResult_2017Dataset_BatchSize_12.PNG) |
| ISIC 2018 | 808 | 0.881 | 0.788 | 0.942 | 0.962 | 0.882 | [2018](results/test_screenshots/TestResult_2018Dataset_BatchSize_12.PNG) |

The same table in the report also lists batch-size-8 runs, which were slightly lower on ISIC 2016 (Dice 0.921, mIoU 0.854) and about the same on ISIC 2017/2018.

**Final team paper (18 May 2025, Table III),** after the architecture and loss changes:

| | ISIC 2016 | ISIC 2017 | ISIC 2018 |
|---|---|---|---|
| Dice | 0.906 | 0.844 | 0.858 |
| mIoU | 0.83 | 0.73 | 0.75 |

The paper states that IoU stays below FAT-Net-class baselines and that the new combined loss did not reliably help.

**Hyperparameter tuning, as shown in the course presentation.** The table has 6 trials, 30 epochs each, on the validation split. The slide does not state which dataset was used; the code default is ISIC 2016.

| Trial | lr | Weight decay | Drop-path | mIoU | DSC (F1) | Accuracy | Specificity | Sensitivity |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.75e-5 | 5.9e-5 | 0.27 | 0.702 | 0.825 | 0.931 | 0.954 | 0.835 |
| 1 | 3.96e-4 | 1.9e-4 | 0.45 | 0.656 | 0.792 | 0.922 | 0.963 | 0.756 |
| 2 | 1.07e-4 | 3.1e-3 | 0.46 | 0.707 | 0.828 | 0.931 | 0.952 | 0.846 |
| 3 | 3.07e-5 | 6.7e-4 | 0.30 | 0.706 | 0.828 | 0.933 | 0.960 | 0.821 |
| 4 | 1.14e-4 | 2.1e-3 | 0.28 | 0.719 | 0.837 | 0.934 | 0.953 | 0.859 |
| **5** | **1.40e-4** | **7.6e-4** | **0.14** | **0.721** | **0.838** | **0.935** | 0.952 | **0.863** |

`optuna_study_report.csv` holds a separate 10-trial run of the same search. It records only the objective (validation loss): the best was 0.339 (trial 0), and the rest ranged from 0.339 to 0.566.

**How to read these results**
- The three tables come from different runs: the best test result is a full 130-epoch run, the paper reports the final configuration, and the tuning table uses short 30-epoch runs on the validation split. They are not directly comparable.
- No checkpoints were kept, so the numbers cannot be re-computed from this repository; they are taken from the report, the paper and the run screenshots.
- No comparison with the unmodified VM-UNet under the same settings was saved, so the effect of the two changes is **not established**. The presentation summarises the cross-dataset results as "same / no change".
- The output paths in the screenshots all say `vmunet_isic16` because the run name in the config was not changed between runs; the test-set sizes (270 / 650 / 808) identify the dataset.

## Running
The environment needs a CUDA GPU with the Mamba kernels. Follow `VM-UNet_README.md` (Python 3.8, PyTorch 1.13, `causal_conv1d`, `mamba_ssm`). Then install the remaining packages, with the versions the VM-UNet authors list:
```bash
pip install -r requirements.txt
```
Then run one of:
```bash
python train.py                    # set datasets = 'isic16' | 'isic17' | 'isic18' in configs/config_setting.py
python train_new.py                # train/val/test split variant (Weights & Biases logging)
python hyperparameter_tuning.py    # Optuna search
```
- I did not re-run training here because the GPU-specific dependencies are missing. All Python files were checked to compile.
- `train_new.py` originally imported the old data loader, which does not accept a `split` argument. It now imports `datasets/dataset_new.py`.

## Repository structure
```
configs/        training configuration (dataset choice, AdvancedLoss, optimiser, scheduler)
datasets/       ISIC loaders (dataset.py original, dataset_new.py split-based)
models/vmunet/  VM-UNet + MultiHeadMamba
engine.py, utils.py (metrics, losses incl. AdvancedLoss)
train.py, train_new.py, hyperparameter_tuning.py, optuna_study_report.csv
*_synapse.py    unchanged upstream scripts for the Synapse dataset
proposal/       project proposal (PDF)
presentation/   project presentations
```

## Academic integrity
This is graded coursework built on an open-source baseline. All credit for the original VM-UNet architecture and code goes to its authors. This repository only adds the changes listed above, made as a team, and does not claim them as individual work.
