project = "GalfitX"
copyright = "2026, Chao Ma"
author = "Chao Ma"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

myst_heading_anchors = 3
master_doc = "index"
source_suffix = {
    ".md": "markdown",
}
