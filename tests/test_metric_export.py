"""Regression check for the real-image export in main.ipynb.

Run with ``python -m unittest discover -s tests`` from the repository root.
Only the export loop runs; no datasets, checkpoints or model weights are needed.
"""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


class RealImageExportTest(unittest.TestCase):
    def test_normalized_rgb_preserves_every_uint8_value(self):
        notebook_path = Path(__file__).resolve().parents[1] / "main.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metric_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "N_METRIC_IMAGES =" in "".join(cell["source"])
        )
        tree = ast.parse(metric_source)
        export_loop = next(
            node
            for node in tree.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and [element.id for element in node.target.elts]
            == ["image_index", "dataset_index"]
        )
        export_code = compile(
            ast.Module(body=[export_loop], type_ignores=[]),
            filename="main.ipynb:real-image-export",
            mode="exec",
        )

        # Each RGB channel contains all 256 possible pixel values four times.
        pixels = np.tile(np.arange(256, dtype=np.uint8), 4).reshape(32, 32)
        expected = np.stack([pixels, np.flipud(pixels), np.fliplr(pixels)], axis=2)
        normalized = torch.from_numpy(expected.copy()).permute(2, 0, 1).float()
        normalized = (normalized / 255.0 - 0.5) / 0.5

        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "metric_real_dataset": [(normalized, 0)],
                "real_indices": [0],
                "real_dir": Path(directory),
                "np": np,
                "Image": Image,
            }
            exec(export_code, namespace)
            with Image.open(Path(directory) / "000000.png") as exported:
                actual = np.asarray(exported)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
