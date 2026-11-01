Weight clipping versus gradient penalty
===========================================

This documentation accompanies the CIFAR-10 comparison of WGAN and WGAN-GP
by Antonio Peña Peña, Geer Wang Liu and Zhenhua Ye Xiang.

Start with :doc:`tutorial` to see how the models turn random vectors into
images, how each training objective updates the networks, and how to interpret
the saved results. The notebook includes code and outputs, so it can also be
read without running Python.

The short optimization example uses synthetic tensors to demonstrate the
training operations. Image generation uses the project's trained checkpoints;
the reported metric values come from the original experiment. The tutorial
labels these sources separately.

.. toctree::
   :maxdepth: 2

   installation
   tutorial
   implementation
   reproducibility

Project materials
---------------------

* `GitHub repository <https://github.com/zhenhuaaaa/Proyecto_IAG>`_
* :download:`Tutorial notebook <tutorial.ipynb>`
* :download:`Original experiment notebook <../main.ipynb>`
* :download:`Project report <../report/IA_GEN_PROYECTO_FINAL.pdf>`
* :download:`Recorded metrics <../evaluation_metrics/metrics_all_checkpoints.csv>`

The notebook follows the explanatory structure requested in the course,
with runnable examples and discussion of their outputs. The
`PyTorch CIFAR-10 tutorial <https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_
is the reference for this format. Sphinx renders the documentation as HTML,
using the notebook's saved outputs.

Scope of the comparison
---------------------------

Both configurations use the same generator architecture and 200,000 generator
updates. Their critics differ in normalization, and their optimizers and
learning rates also differ. The results therefore compare these configurations
as a whole. They do not isolate the effect of gradient penalty.

The documentation preserves the original measurements. A preprocessing
correction in the real-image PNG export affects future evaluation runs;
:doc:`reproducibility` explains the change and the provenance limitations.
