# Weight clipping versus gradient penalty in Wasserstein GAN training

This project compares WGAN with weight clipping and WGAN-GP on CIFAR-10. Both methods generate 32 × 32 RGB images from Gaussian noise. The comparison covers generated samples, training losses, critic input gradients, FID, KID, Inception Score, Recall and Coverage.

Authors: Antonio Peña Peña, Geer Wang Liu and Zhenhua Ye Xiang.

Repository: [zhenhuaaaa/Proyecto_IAG](https://github.com/zhenhuaaaa/Proyecto_IAG).

The course submission consists of this repository URL and the project report PDF. The tutorial and documentation are part of the repository.

## Start with the tutorial

Open [the tutorial notebook](docs/tutorial.ipynb) to follow the implementation step by step. It explains the data representation, model architectures, loss functions and optimization steps, then loads trained generators and discusses the saved experimental results. The notebook contains outputs for reading without execution.

The default tutorial runs a small optimization demonstration on synthetic tensors, generates images from the supplied checkpoints and plots the recorded metrics. The synthetic demonstration checks the training operations; it does not train a useful image generator. Downloading CIFAR-10 is an optional step. The full experiment remains in [main.ipynb](main.ipynb).

The [project report](report/IA_GEN_PROYECTO_FINAL.pdf) presents the research question, methodology and discussion. For function signatures, checkpoint formats and complete reproduction instructions, read the documentation built with Sphinx.

## Run the tutorial

Use Python 3.11 and Git LFS. From a terminal:

```bash
git clone https://github.com/zhenhuaaaa/Proyecto_IAG.git
cd Proyecto_IAG
git lfs install
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab docs/tutorial.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. If the project is already downloaded, start from its directory and create the environment there. A GitHub ZIP may contain Git LFS pointer files instead of the WGAN-GP weights; use a clone with `git lfs pull` for checkpoint inference.

Select the environment's Python kernel, then run the tutorial cells in order. Do not use `Run All` on `main.ipynb` as an introductory step: it includes long training loops, checkpoint writes, dataset downloads and full metric recomputation.

## Read or rebuild the HTML documentation

Read the tutorial directly on GitHub using the link above. To read the Sphinx HTML locally, build it and open `docs/_build/html/index.html` in a browser.

To build from source, run these commands from the repository root:

```bash
python -m pip install -r requirements-docs.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Sphinx renders the notebook's stored outputs. Building the documentation does not start training or recalculate metrics. See [installation](docs/installation.rst) for separate reading, execution and evaluation requirements.

The tutorial has been executed from a fresh kernel, and the HTML build passes with warnings treated as errors. [Validation notes](VALIDATION.md) record the checks and their scope.

## Recorded results

The table below reproduces [the original metrics CSV](evaluation_metrics/metrics_all_checkpoints.csv) at 200,000 generator updates. These are stored experimental results, not measurements from the tutorial's short demonstration.

| Model | FID ↓ | KID ↓ | Inception Score ↑ | Recall ↑ | Coverage ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| WGAN | 42.7442 | 0.026846 | 6.1363 | 0.4268 | 0.4549 |
| WGAN-GP | 40.1998 | 0.026540 | 6.2666 | 0.4295 | 0.5176 |

WGAN-GP has better values for these five metrics at the final checkpoint, with a larger advantage earlier in training. The experiment also changes the critic normalization, optimizer and learning rate, so it cannot attribute the differences to gradient penalty alone. A single training run per configuration does not establish statistical significance.

## Repository contents

| Path | Contents |
| --- | --- |
| `docs/tutorial.ipynb` | Guided, executable tutorial with saved outputs |
| `docs/` | Sphinx sources, installation guide, implementation reference and reproduction guide |
| `main.ipynb` | Original training and evaluation experiment |
| `checkpoints/wgan_bn/` | WGAN checkpoints used in the reported comparison |
| `checkpoints/wgan_gp/` | WGAN-GP checkpoints, stored through Git LFS |
| `checkpoints/wgan/` | Additional WGAN checkpoints without critic BatchNorm; not the main comparison |
| `evaluation_metrics/` | Recorded metric table and evaluation image files |
| `results/` | Saved sample grids from the experiments |
| `references/` | Reference implementations, separate from the project's PyTorch notebook |
| `report/IA_GEN_PROYECTO_FINAL.pdf` | Project manuscript |
| `tests/test_metric_export.py` | Regression check for normalized-image PNG export |

## Reproduction notes

The tutorial and documentation distinguish recorded measurements from newly executed examples. Training uses 50,000 CIFAR-10 training images; evaluation uses the separate test split. The original notebook uses `cycle(train_loader)`, which repeats the batches cached during its first pass. It does not save random generator state for an exact continuation of training.

The real-image export in `main.ipynb` now reverses the `[-1, 1]` normalization before converting images to PNG. The previous code assumed `[0, 1]` inputs. The recorded CSV and its original outputs are preserved; they have not been recomputed as part of writing the documentation. See the [reproduction guide](docs/reproducibility.rst) for the preprocessing correction and the limits of the available provenance.

To recalculate FID, KID, Inception Score, Recall and Coverage, install `requirements-eval.txt` and follow that guide. Full evaluation loads pretrained Inception weights and processes 10,000 real and 10,000 generated images per checkpoint.
