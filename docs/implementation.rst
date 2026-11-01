Implementation reference
============================

The experiment is implemented in ``main.ipynb`` at the repository root. Its
classes and functions are notebook definitions, rather than an installable
Python package. The :doc:`tutorial` explains and executes the core operations;
this page records their interfaces and the file formats needed to reuse them.
See :doc:`reproducibility` for the order in which to run the full experiment.

Models and tensor shapes
----------------------------

``Generator(latent_dim=128, dim=128)`` maps independent standard Gaussian
vectors of shape ``(B, 128)`` to RGB tensors of shape ``(B, 3, 32, 32)``.
Its final ``Tanh`` restricts image values to ``[-1, 1]``. Both methods use
this architecture, but train separate instances with different weights.

.. list-table:: Generator with the experiment's default dimensions
   :header-rows: 1
   :widths: 55 45

   * - Operation
     - Output shape
   * - Linear, batch normalization, ReLU
     - ``(B, 8192)``
   * - Reshape
     - ``(B, 512, 4, 4)``
   * - Transposed convolution, batch normalization, ReLU
     - ``(B, 256, 8, 8)``
   * - Transposed convolution, batch normalization, ReLU
     - ``(B, 128, 16, 16)``
   * - Transposed convolution, Tanh
     - ``(B, 3, 32, 32)``

``Critic(dim=128)`` and ``CriticGP(dim=128)`` map an image batch to scores
of shape ``(B, 1)``. Scores are real numbers, not probabilities: neither
critic uses a sigmoid. Their three convolutional stages produce
``(B, 128, 16, 16)``, ``(B, 256, 8, 8)`` and ``(B, 512, 4, 4)``.
Flattening and a linear layer produce the score.

All convolutions use a ``5 x 5`` kernel, stride 2 and padding 2.
Transposed convolutions also use ``output_padding=1``. Critic activations
are ``LeakyReLU(0.2)``. ``Critic`` applies batch normalization after its
second and third convolutions; ``CriticGP`` has no normalization layers.
This avoids dependencies between samples when computing its gradient penalty.

The generator's batch normalization requires more than one sample when
training. For inference, call ``model.eval()`` and use ``torch.no_grad()``.
Convert generated tensors to display values using
``((images + 1) / 2).clamp(0, 1)`` and permute channels from CHW to HWC.

Training functions
----------------------

Both methods minimize the following critic component and generator loss:

.. math::

   L_D = \operatorname{mean}(D(G(z))) - \operatorname{mean}(D(x)),
   \qquad L_G = -\operatorname{mean}(D(G(z))).

The sign of a loss is not a classification accuracy or a direct measure of
image quality. In particular, WGAN-GP's total critic loss also contains a
regularization term and must not be confused with its Wasserstein component.

``train_critic(generator, critic, real_images, optimizer_D, z_dim, device)``
   Samples a fresh latent batch, detaches generated images from the generator
   graph, minimizes ``L_D``, and clips every trainable critic parameter to
   ``[-0.01, 0.01]`` after the optimizer step. This includes the affine batch
   normalization parameters. Returns the pre-update loss as a Python float.

``train_generator(generator, critic, optimizer_G, batch_size, z_dim, device)``
   Samples a fresh latent batch, minimizes ``L_G`` and steps only
   ``optimizer_G``. Returns the pre-update loss as a Python float. Gradients
   flow through the critic to the generated images; its optimizer is not
   stepped during this operation.

``interpolate_samples(real_images, fake_images)``
   Expects matching ``(B, 3, 32, 32)`` tensors on the same device. Samples
   one coefficient per image, with shape ``(B, 1, 1, 1)``, and returns
   ``real_images + alpha * (fake_images - real_images)`` with the same shape.

``gradient_penalty(critic, interpolated_images)``
   Detaches interpolations, enables input gradients, and computes a scalar
   tensor equal to the mean squared deviation of input-gradient norms from 1.
   Each norm spans all 3,072 input components. ``create_graph=True`` allows
   backpropagation through this derivative into the critic parameters.

.. math::

   \mathrm{GP} = \frac{1}{B}\sum_{i=1}^{B}
   \left(\left\|\nabla_{\hat{x}_i}D(\hat{x}_i)\right\|_2 - 1\right)^2.

``wgan_gp_critic_loss(critic, real_images, fake_images, lambda_gp=10.0)``
   Returns three scalar tensors in order: ``L_D + lambda_gp * GP``,
   ``L_D`` and the unweighted ``GP``. Generated inputs are detached.

``train_critic_gp(generator, critic, real_images, optimizer_D, z_dim, device, lambda_gp=10.0)``
   Generates images under ``torch.no_grad()``, computes the three losses,
   and updates the critic using their total. It does not clip weights.
   Returns total loss, Wasserstein component and unweighted penalty as floats.

``train_generator_gp(generator, critic, optimizer_G, batch_size, z_dim, device)``
   Uses the same generator objective as ``train_generator`` and returns a
   float. The two functions have separate names in the original notebook.

``detach()`` and ``torch.no_grad()`` prevent gradient propagation to the
generator during critic optimization. They do not switch the generator to
evaluation mode: its batch normalization running statistics can still change.
Likewise, the WGAN critic remains in training mode during generator updates.

Experiment configuration
----------------------------

.. list-table:: Configuration in ``main.ipynb``
   :header-rows: 1
   :widths: 48 26 26

   * - Setting
     - WGAN
     - WGAN-GP
   * - Batch size / latent dimension / base channels
     - 64 / 128 / 128
     - 64 / 128 / 128
   * - Critic updates per generator update
     - 5
     - 5
   * - Generator update budget
     - 200,000
     - 200,000
   * - Optimizer for both networks
     - RMSprop
     - Adam
   * - Learning rate
     - ``5e-5``
     - ``1e-4``
   * - Adam betas
     - Not applicable
     - ``(0.5, 0.9)``
   * - Constraint parameter
     - Clipping threshold ``0.01``
     - Penalty weight ``10.0``
   * - Recovery checkpoint interval
     - 2,500 generator updates
     - 1,000 generator updates

A reported iteration is a generator update, with five critic updates per
iteration. The complete budget is therefore one million critic updates per
method. The comparison uses checkpoints at 1,000, 10,000, 100,000 and 200,000
generator updates.

Checkpoint formats
----------------------

WGAN uses ``checkpoints/wgan_bn/``. The separate ``checkpoints/wgan/``
directory contains another variant and is not the WGAN baseline selected by
the comparison in section 5 of ``main.ipynb``.

.. list-table:: Saved dictionary keys
   :header-rows: 1
   :widths: 30 35 35

   * - Content
     - WGAN
     - WGAN-GP
   * - Generator weights
     - ``generator``
     - ``generator_state_dict``
   * - Critic weights
     - ``critic``
     - ``critic_state_dict``
   * - Optimizer states
     - ``optimizer_G``, ``optimizer_D``
     - ``optimizer_G_state_dict``, ``optimizer_D_state_dict``
   * - Loss histories
     - ``generator_losses``, ``critic_losses``
     - Same, plus ``wasserstein_losses``, ``gradient_penalties``
   * - Update counter
     - ``iteration``
     - ``iteration``
   * - Extra configuration
     - Use the notebook constants
     - ``z_dim``, ``dim``, ``lambda_gp``, ``critic_iterations``, ``learning_rate``, ``betas``, ``fixed_noise``

WGAN's ``save_wgan_checkpoint(iteration, path)`` uses global model, optimizer
and history variables. Its recovery file ``checkpoint.pth`` stores the
zero-based index of the last completed update; the loader adds 1 before
resuming. Milestone files such as ``wgan_iter_200000.pth`` instead store the
number of completed updates. They are evaluation snapshots and must not be
renamed to ``checkpoint.pth`` without adjusting the counter convention.

WGAN-GP's ``save_wgan_gp_checkpoint(...)`` takes models, optimizers, histories
and fixed noise as explicit arguments, but reads configuration constants from
the notebook. Both its recovery file ``wgan_gp_latest.pt`` and milestones
``wgan_gp_iter_200000.pt`` store the number of completed generator updates.
Its loader resumes with ``range(saved_iteration, MAX_ITERATIONS_GP)``.

These counter conventions describe the current source. The historical files
have discrepancies: ``wgan_iter_200000.pth`` stores ``iteration=199999`` with
200,001 generator and 1,000,005 critic loss entries. The WGAN-GP milestone
stores ``iteration=200000`` with 200,034 generator and 1,000,193 critic
entries. The files do not reveal which earlier execution or interruption
created these differences. The tutorial therefore plots each loss history
against its own recorded index instead of assuming exact alignment.

Checkpoints do not store the Python, NumPy or PyTorch random-number-generator
states, or the data iterator position. Resuming restores weights and optimizer
state but does not reproduce an uninterrupted run exactly. An interrupted
WGAN-GP iteration may also have recorded partial critic updates.

Evaluation helpers
----------------------

``load_checkpoint(path)``
   Reads a checkpoint on ``DEVICE`` and reports a missing path explicitly.
   The original loader uses ``weights_only=False``; use it only with these
   project checkpoints or other files whose origin you trust.

``get_generator_state(ckpt)``
   Accepts generator weights under ``generator_state_dict``, ``generator``
   or ``generator_state``.

``load_generator(model_name, iteration)``
   Resolves ``CHECKPOINTS``, checks the latent dimension, loads the generator
   and switches it to evaluation mode. Returns model, checkpoint dictionary
   and saved iteration. Section 5 retains all eight models and full
   checkpoint dictionaries in ``loaded``, which requires substantial RAM.

``moving_average(values, window=200)``
   Returns x coordinates and the valid-window moving average. Empty inputs
   produce empty arrays. Histories shorter than the window are returned
   unsmoothed. The loss plot divides critic-update positions by five to align
   them with generator-update positions.

``get_training_history(model_name, iteration)``
   Returns generator, critic, Wasserstein-component and gradient-penalty
   histories from ``loaded``. Missing histories become empty lists.

``gradient_norms_for_checkpoint(model_name, iteration, dataset, n_images=1000, batch_size=64)``
   Copies the corresponding critic template, loads its weights and switches
   it to evaluation mode. It selects real images using seed 42 and returns
   one NumPy input-gradient norm per image. These are gradients at real test
   images, not the interpolated inputs used by the training penalty. Keep
   autograd enabled when calling this function.

``save_generated_images(generator_model, output_dir, n_images, seed)``
   Generates batches from a locally seeded CPU noise generator, converts the
   output from ``[-1, 1]`` to ``[0, 1]``, and writes numbered RGB PNG files.
   It deletes existing PNGs in ``output_dir`` before writing.

``inception_features_from_folder(folder, batch_size=64)``
   Creates an ImageFolder wrapper using hard links or copies, resizes images
   to ``299 x 299``, applies ImageNet normalization, and returns features
   from the global torchvision Inception-v3 model with its final classifier
   replaced by ``nn.Identity``. These features feed Recall and Coverage.
