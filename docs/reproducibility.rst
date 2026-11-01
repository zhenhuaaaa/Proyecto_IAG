Running and reproducing the experiment
==========================================

Start with the :doc:`tutorial`. It is the guided demonstration for readers
and can be read with its saved outputs. The full experiment in ``main.ipynb``
trains two networks for 200,000 generator updates each and recomputes metrics
for eight checkpoints. Running every cell is a separate, costly operation.

Installation and files
--------------------------

Use Python 3.11 in a virtual environment. Run the following commands from the
directory in which you want to keep the project:

.. code-block:: console

   git clone https://github.com/zhenhuaaaa/Proyecto_IAG.git
   cd Proyecto_IAG
   git lfs pull
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m jupyter lab

On Windows, activate the environment with ``.venv\Scripts\activate``.
Git LFS must be installed before ``git lfs pull``; the repository tracks
``*.pt`` through LFS. A tiny text file beginning with
``version https://git-lfs.github.com/spec/v1`` is a pointer, not a usable
checkpoint. Recover its actual content before inference. The WGAN files
use the ``.pth`` extension and the WGAN-GP files use ``.pt``.

The CPU build is sufficient for the short tutorial. For a full training run,
install a mutually compatible PyTorch and torchvision pair for your hardware
and choose the device explicitly in the first cell of ``main.ipynb``. Its
saved source sets ``DEVICE = torch.device("cpu")``. To use CUDA:

.. code-block:: python

   DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

All paths in ``main.ipynb`` are relative to the repository root. Launch
Jupyter there and keep the notebook there. The key inputs and outputs are:

.. list-table:: Repository map
   :header-rows: 1
   :widths: 42 58

   * - Path
     - Purpose
   * - ``main.ipynb``
     - Original training and evaluation experiment
   * - ``docs/tutorial.ipynb``
     - Explained, short demonstration with saved outputs
   * - ``checkpoints/wgan_bn/``
     - WGAN baseline with critic batch normalization
   * - ``checkpoints/wgan_gp/``
     - WGAN-GP weights, optimizer states and histories
   * - ``evaluation_metrics/metrics_all_checkpoints.csv``
     - Historical measurements from the project
   * - ``evaluation_metrics/real/`` and model-specific subdirectories
     - PNG inputs to the original metric calculation
   * - ``results/``
     - Additional saved image grids
   * - ``references/``
     - Reference implementations, not the executable entry point
   * - ``data/``
     - Downloaded CIFAR-10 files, excluded from Git

The full notebook downloads CIFAR-10 when its data cells run. It uses 50,000
training images with labels ignored, and a separate test split of 10,000
images for evaluation. Training inputs are normalized from ``[0, 1]`` to
``[-1, 1]`` with mean and standard deviation both equal to 0.5.

Using the saved checkpoints
-------------------------------

The tutorial already provides inference without retraining. To reproduce
the original section-5 figures from ``main.ipynb`` in a fresh kernel, run
only the following definitions and evaluation cells, in order. Cell indices
below are zero-based JSON positions, not Jupyter execution counters.

1. Run the imports and configuration cell 0, then the transform definition
   in cell 2. Set the device before proceeding.
2. Run the ``Generator`` definition in section 2.2, cell 11; ``Critic`` in
   section 2.3, cell 13; and ``CriticGP`` at the start of section 4, cell 18.
3. Create the critic templates in a new scratch cell without running training:

   .. code-block:: python

      critic = Critic(dim=MODEL_DIM).to(DEVICE)
      critic_gp = CriticGP(dim=MODEL_DIM).to(DEVICE)

4. Run the checkpoint mapping and loading cell 35 under section 5. It uses
   ``checkpoints/wgan_bn/`` for WGAN and ``checkpoints/wgan_gp/`` for WGAN-GP.
5. Run cells 37, 39 and 41 for the image comparison, loss histories and
   gradient-penalty curve. They use saved weights and histories.
6. For section 5.4, run cells 42, 44 and 45. The last cell downloads the
   test split if necessary and evaluates input gradients on 1,000 real
   images per checkpoint.
7. To plot the existing metric CSV, run cell 48 directly. Cell 47 is the
   expensive metric recomputation, described below; it is not needed to
   read or plot existing values.

The image comparison uses 64 shared latent vectors with seed 42. The
gradient-norm helper independently uses the same seed to choose its real
test subset, so the subset is shared across checkpoint comparisons. In
evaluation mode, batch normalization uses stored running statistics.

Starting or resuming training
---------------------------------

For a new experiment, first make a separate working copy or change both
checkpoint directory settings to fresh directories. The notebook automatically
loads recovery files when they exist, and saves results into those same paths.
Keep the supplied checkpoints available for the documented comparison.

Run the first configuration cell, section 1 and the model definitions in
section 2. The data loader uses batch size 64, ``shuffle=True`` and
``drop_last=True``. Then choose the desired training section:

* For WGAN, run both cells under section 3: the update functions and the
  combined initialization, recovery and training cell.
* For WGAN-GP, run section 4 through its training loop. It defines the critic,
  interpolation and penalty, update functions, configuration, optimizers,
  checkpoint saving and recovery before starting the loop.

For a small development run, set ``N_ITERATIONS`` to a small value in the
first cell and use fresh checkpoint directories. WGAN-GP copies this value
to ``MAX_ITERATIONS_GP``. If you also want the original evaluation cells to
consume those checkpoints, update both ``CHECKPOINT_ITERATIONS`` and the
hard-coded ``CHECKPOINTS`` dictionary to the new milestones and paths. A short
run does not reproduce the reported trained models or metrics.

WGAN recovers from ``checkpoints/wgan_bn/checkpoint.pth``; WGAN-GP recovers
from ``checkpoints/wgan_gp/wgan_gp_latest.pt``. Their counter conventions
differ, as detailed in :doc:`implementation`. WGAN-GP also saves a snapshot
when its training loop catches an interruption or exception. WGAN only has
its periodic recovery saves. Both save permanent comparison milestones.

The original implementation uses ``itertools.cycle(train_loader)``. It
caches the first pass through the loader, then repeats those same batches
without a new shuffle. This also retains the first pass in memory. A new
implementation that reshuffles every pass is a changed experiment, even if
the models and optimizers are unchanged.

Seeds for evaluation do not seed the original training run. The supplied
training code does not record the initial random states, and the checkpoints
do not preserve random states or loader position. Reusing a checkpoint
reproduces inference from its weights; rerunning training is not guaranteed
to reproduce those weights or the CSV exactly.

Recomputing quantitative metrics
------------------------------------

Install the optional evaluation dependencies:

.. code-block:: console

   python -m pip install -r requirements-eval.txt

Complete the checkpoint-loading and test-dataset steps above, then run cell
47 under section 5.5. It generates 10,000 images for each of the eight
model/checkpoint combinations and compares them with all 10,000 test images.
The pretrained Inception weights require a download unless already cached.
This computation can require substantial memory, disk space and time,
particularly the pairwise distances used by PRDC.

Cell 47 deletes existing PNGs in its target image directories and replaces
``metrics_all_checkpoints.csv`` after evaluation. To preserve the submitted
measurements, use a separate output directory, for example by changing
``METRIC_DIR`` to ``Path("evaluation_metrics_recomputed")``. Point the plotting
cell to its CSV if you want to visualize the recomputed values.

The metric configuration is:

* ``METRIC_SEED = 1234`` controls the NumPy selection of real images and the
  locally seeded generator used to sample fake images. All 10,000 test images
  are included, in a permuted order.
* ``torch_fidelity.calculate_metrics`` computes FID, KID and Inception Score
  from PNG folders using its default feature extractor. KID uses five subsets
  of 1,000 samples. The notebook does not explicitly pass a ``rng_seed`` to
  this library; its internal sampling follows the installed library's default.
* Recall and Coverage use ``prdc.compute_prdc`` with ``nearest_k=5`` and
  features extracted separately with torchvision's pretrained Inception-v3.
  These features are not the same preprocessing pipeline as torch-fidelity's
  metric features.
* The torchvision pipeline resizes to ``299 x 299``, normalizes with ImageNet
  mean and standard deviation, uses ``transform_input=False``, and replaces
  the classifier with ``nn.Identity()``.

.. important::

   The supplied notebook originally exported normalized real tensors by
   multiplying them directly by 255 and converting to ``uint8``. Since those
   tensors are in ``[-1, 1]``, this omits denormalization and can wrap negative
   values. Real and generated inputs must both be converted to ``[0, 1]``
   before PNG export. The documentation work corrects the real export code;
   the existing CSV and manuscript values have not been recalculated. Treat
   them as historical reported measurements, not measurements validated by
   this correction. Recompute the metrics with the corrected export before
   relying on their absolute values or claiming exact reproduction.

For a normalized RGB tensor ``image``, the correct export is:

.. code-block:: python

   image = ((image + 1) / 2).clamp(0, 1)
   array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

The known defect affects metrics that compare generated images with a real
reference, namely FID, KID, Recall and Coverage. Inception Score uses only
generated images, so that specific real-export defect does not change its
inputs. A small-sample test run can check the code path, but must not replace
the 10,000-image experiment under the same result label.

Interpreting the outputs
----------------------------

The historical CSV has eight rows. At 200,000 iterations it records FID
42.7442 versus 40.1998, KID 0.0268463 versus 0.0265401, Inception Score
6.1363 versus 6.2666, Recall 0.4268 versus 0.4295, and Coverage 0.4549 versus
0.5176 for WGAN and WGAN-GP respectively. These are read from the supplied
file; the tutorial does not rerun the metric networks to obtain them.

Several limits matter when reading these results:

* The methods differ in critic normalization, optimizer and learning rate,
  as well as clipping versus gradient penalty. The comparison does not
  isolate the causal effect of the penalty.
* There is no collection of independent training seeds. KID and Inception
  Score standard deviations in the CSV are internal estimator summaries,
  not uncertainty across training runs.
* Gradient-norm histograms use real test inputs, whereas the penalty acts on
  interpolations. A low penalty does not establish a global 1-Lipschitz bound.
* Historical checkpoint counters and loss-history lengths differ. The final
  WGAN file has 200,001 generator loss records and the WGAN-GP file has
  200,034, despite filenames labelled 200,000. The tutorial uses recorded
  indices for loss plots; exact alignment cannot be recovered from these logs.
* Critic losses have different scales, and WGAN-GP's total loss includes
  regularization. Compare its stored Wasserstein component when discussing
  critic training; use images and validated metrics to assess sample quality.

Building the documentation
------------------------------

The documentation source is in ``docs/`` and uses Sphinx with MyST-NB.
The build reads the tutorial's saved outputs without retraining or
recomputing metrics:

.. code-block:: console

   python -m pip install -r requirements-docs.txt
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. The notebook remains
available as ``docs/tutorial.ipynb`` for inspection and execution in Jupyter.
See the :doc:`index` for the reading order and repository link.
