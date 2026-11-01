Installation and reading options
====================================

Read the delivered documentation
------------------------------------

Open ``docs/tutorial.ipynb`` in the GitHub repository to read the explanations,
code, figures and saved outputs without installing Python. The course submission
consists of the repository URL and the report PDF; the documentation lives in
the repository. The build instructions below produce an optional local HTML copy.

The notebook is also available as :download:`tutorial.ipynb <tutorial.ipynb>`.
Keep its saved outputs when viewing it in Jupyter or GitHub. Re-execution uses
the repository's checkpoint files, which are not embedded in the notebook.

Set up the execution environment
------------------------------------

The dependency files target Python 3.11. They specify a compatible PyTorch and
torchvision pair for the tutorial; they are not a record of the original
training environment. The original experiment did not include a complete
dependency lock file.

Clone the repository and obtain the Git LFS objects:

.. code-block:: bash

   git clone https://github.com/zhenhuaaaa/Proyecto_IAG.git
   cd Proyecto_IAG
   git lfs install
   git lfs pull

If the project is already cloned, run the last two commands in its directory.
Files ending in ``.pt`` are tracked with Git LFS. A small text file beginning
with ``version https://git-lfs.github.com/spec/v1`` is a pointer, not a model
checkpoint. Download its LFS object before running inference. The additional
``.pth`` files are ordinary repository files.

Create an environment and install the tutorial dependencies:

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m jupyter lab docs/tutorial.ipynb

On Windows, use ``py -3.11 -m venv .venv`` and activate the environment with
``.venv\Scripts\activate``. Select the Python kernel from this environment.

Run the tutorial cells from the beginning. Its normal path uses CPU and the
supplied checkpoints. The synthetic optimization example is deliberately
small and does not replace CIFAR-10 training. The CIFAR-10 loading section
explains how to enable the optional dataset download.

The original ``main.ipynb`` contains the complete experiment and resumes
training from existing recovery checkpoints. Read :doc:`reproducibility`
before executing it because some cells write checkpoints, export image folders
or recalculate metrics.

Dependency groups
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Purpose
   * - ``requirements.txt``
     - Execute the tutorial and the core PyTorch experiment.
   * - ``requirements-docs.txt``
     - Build the Sphinx HTML documentation from saved notebook outputs.
   * - ``requirements-eval.txt``
     - Add torch-fidelity, PRDC and their numerical dependencies for full metric evaluation.

Full metric evaluation also requires the CIFAR-10 test set and pretrained
Inception weights. Their first download needs network access. The tutorial's
plots of the stored CSV do not require these metric packages.

Build the HTML documentation
--------------------------------

From the repository root:

.. code-block:: bash

   python -m pip install -r requirements-docs.txt
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` after a successful build. Sphinx uses
MyST-NB to render ``docs/tutorial.ipynb`` and its stored outputs. Notebook
execution is disabled during the build, so generating the HTML does not
download CIFAR-10, update model weights or recompute metrics.

To update the example outputs, execute and save the tutorial in Jupyter,
then rebuild the HTML. Keep the notebook beside the other documentation
sources so its relative paths continue to resolve from the repository.

Troubleshooting
-------------------

If an import fails, confirm that the Jupyter kernel belongs to the environment
where the requirements were installed. Keep the PyTorch and torchvision
versions paired; mixing independently installed versions can fail when
torchvision registers its operators.

If a checkpoint cannot be read, check the file exists and is not an LFS
pointer. Open the tutorial from the repository checkout, which supplies
``checkpoints/`` and ``evaluation_metrics/``. A standalone downloaded notebook
contains its saved figures but does not contain the trained models.

If a Sphinx command reports a missing file, check that the tutorial, report
and metrics CSV are present. Use the repository root as the working directory
for the documented build command.
