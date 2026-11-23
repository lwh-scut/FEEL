# FEEL

FEEL (fNIRS–EEG Emotion Library) is an open-source PyTorch library and benchmark built on LibEER. It supports training, evaluating, and comparing models for music emotion recognition from EEG, functional near-infrared spectroscopy (fNIRS), and paired EEG–fNIRS recordings. The training scripts classify valence, arousal, and dominance as separate binary tasks.

## Installation

From the repository root, run the following in a Python 3.10 or newer environment.

```bash
python -m pip install -e .
```

Training uses CUDA when PyTorch detects it and otherwise selects CPU. Use `--device` to select a device explicitly. The CUDA build of PyTorch and its matching TorchAudio build must be installed in the environment for GPU training.

## Training

```bash
python -m feel.train.eeg.train_eegnet --eeg-dir data/recordings --labels data/labels.xlsx --strategy single-subject --label-dimension valence
python -m feel.train.fnirs.train_dbjnet --fnirs-dir data/recordings --labels data/labels.xlsx
python -m feel.train.eeg_fnirs.train_tcresnet --eeg-dir data/eeg --fnirs-dir data/fnirs --labels data/labels.xlsx
```

Run a training module with `--help` for its supported splits and options. By default, each main entry point runs its original set of splits for all three emotion dimensions. `--epochs` overrides the model's default training length. Results go to `runs/<modality>/<model>/` by default. Use `--output-dir` to set a different destination. The scripts retain the original spelling `cross-trail` for the cross-trial split.

### Models with training entry points

| Modality | Training package | Model names used in module filenames |
| --- | --- | --- |
| EEG | `feel.train.eeg` | `acrnn`, `cdcn`, `coral_dgcnn`, `dann_dgcnn`, `dbn`, `dgcnn`, `eegnet`, `fbstcnet`, `gcbnet`, `gcbnet_bls`, `nsal_dgat`, `rgnn`, `tsception` |
| fNIRS | `feel.train.fnirs` | `dbjnet`, `fnirs_net`, `fnirs_transformer`, `hsctn` |
| EEG–fNIRS | `feel.train.eeg_fnirs` | `e_fnet`, `eca_fusion_net`, `efnet`, `fganet`, `gcn_ca_capsnet`, `m2nn`, `tcresnet`, `tsmmf` |

Prefix the model name with `train_` when invoking a training module. The `e_fnet` and `efnet` modules are separate implementations.

## Input data

Supply preprocessed MATLAB `.mat` recordings and an Excel `.xlsx` workbook. A directory can contain both EEG and fNIRS files, or each modality can have its own root. The example below uses a shared recording directory.

```text
data/
  labels.xlsx
  recordings/
    song01/
      sub-001/
        recording_EEG.mat
        recording_NIRS.mat
```

- The loaders extract the song and subject IDs from the `song<number>` and `sub-<number>` path components. The workbook needs `music_id` and `sub_id` columns containing matching numeric IDs, plus a column for each requested dimension (`valence`, `arousal`, `dominance`).
- EEG filenames contain `EEG`. Their MAT files contain a channels-by-time `data` array and `srate`. The loader takes the last 1,500 samples or pads shorter recordings with zeros on the left.
- fNIRS filenames contain `NIRS`. Their MAT files contain a channels-by-time `data` array and a `time` vector. The loader takes the last 150 samples or pads shorter recordings with zeros on the left.
- The default training inputs use 32 EEG channels and 40 fNIRS channels. Paired training matches recordings by subject and song. The loaders map ratings above 5 to class 1 and ratings at or below 5 to class 0.

## Repository layout

| Directory | Contents |
| --- | --- |
| `feel/models/` | Model implementations organized by modality and the model registry |
| `feel/train/` | Main training entry points organized by modality |
| `feel/data/` | Recording loaders, preprocessing, and data splits |
| `feel/training/` | Training and evaluation loops |
| `feel/utils/` | Metrics and result export |
| `feel/experiments/` | Earlier fNIRS presets and EEG experiments for other input formats |

Model classes can also be imported directly, for example `from feel.models.eeg.eegnet import EEGNet`.

The earlier fNIRS scripts use `python -m feel.experiments.legacy_fnirs.train_<model>` with `--fnirs-dir` and `--labels`. The EEG experiments are `feel.experiments.train_eeg_json` for JSON epochs with per-trial `label.txt` files and `feel.experiments.train_eeg_imagery` for direction-specific NPY files named `<subject-id>_imagery_<direction>_data.npy`. Run either module with `--help` for its input arguments.
