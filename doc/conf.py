project = "GalfitX"
copyright = "2026, Chao Ma"
author = "Chao Ma"

extensions = [
    "myst_parser",
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

# Use web/ as additional source directory
import os
html_context = {}
