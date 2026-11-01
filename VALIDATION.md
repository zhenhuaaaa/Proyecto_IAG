# Documentation validation

Validated on 1 November 2026 using Python 3.11.7 on CPU.

- Executed all 19 code cells in `docs/tutorial.ipynb` from a fresh Jupyter kernel. Execution finished without cell errors in approximately 56 seconds on the validation machine. This is not an estimate of training time.
- Checked tensor shapes and ranges, normalization round-trip, an analytic gradient penalty, finite optimization losses, generator weight updates and the WGAN clipping bound.
- Loaded both supplied final generator checkpoints with strict state-dictionary matching and generated 16 images per model using shared noise.
- Read the actual checkpoint histories and original eight-row metric CSV. The tutorial explains discrepancies between saved counters and history lengths.
- Confirmed that all nine copied implementation cells match their corresponding definitions in `main.ipynb`.
- Passed `python -m unittest discover -s tests -v`. The regression test runs the real notebook export loop and preserves all 256 values per RGB channel. The same test failed before the denormalization correction.
- Built HTML with `python -m sphinx -W --keep-going -b html docs docs/_build/html`, with no warnings.
- Checked 189 local HTML links and asset references; all target files exist. The historical notebook figures are embedded as attachments, and MathJax is included locally for offline equations.

The validation environment uses the package versions recorded in the notebook's first output and the requirements files. These versions do not reconstruct the original training environment.

The generated sample PNG was inspected directly. Chrome in the validation environment aborted local-page navigation, so a browser screenshot could not be verified; the HTML checks above cover compilation and local resource paths.

The optional CIFAR-10 download branch, full training and full FID/KID/IS/Recall/Coverage recomputation were not completed during this documentation validation. The historical metric CSV, report, checkpoints and original notebook outputs remain unchanged. Only the original notebook's real-image export code received the documented normalization correction.
