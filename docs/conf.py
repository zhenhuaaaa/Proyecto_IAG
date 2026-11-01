project = "WGAN and WGAN-GP on CIFAR-10"
author = "Antonio Peña Peña, Geer Wang Liu, Zhenhua Ye Xiang"
copyright = "2026, " + author
extensions = ["myst_nb", "sphinx.ext.mathjax"]
root_doc = "index"
exclude_patterns = ["_build", ".ipynb_checkpoints", "**/.ipynb_checkpoints"]
nb_execution_mode = "off"  # Publish the verified outputs; never retrain during a build.
nb_execution_raise_on_error = True
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3
html_theme = "alabaster"
html_title = "WGAN versus WGAN-GP | Project tutorial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {"description": "Implementation, experiments and reproducibility", "fixed_sidebar": True}
html_sidebars = {"**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]}

# Use the vendored MathJax bundle so equations also render without internet.
mathjax_path = "mathjax/tex-chtml.js"


def resolve_notebook_attachments(app, docname, source):
    """Keep portable notebook attachments and use their local copies in HTML."""
    if docname == "tutorial":
        for name in ("cifar10_training_preview", "historical_samples", "historical_gradient_norms"):
            source[0] = source[0].replace(f"attachment:{name}.png", f"_static/{name}.png")


def setup(app):
    app.connect("source-read", resolve_notebook_attachments)
