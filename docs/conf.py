"""Sphinx configuration for sustainable-jsp documentation."""

import os
import sys

# Make the package importable without installing
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "sustainable-jsp"
author = "sustainable-jsp contributors"
copyright = "2026, sustainable-jsp contributors"
release = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",           # extract docstrings automatically
    "sphinx.ext.napoleon",          # support NumPy & Google docstring style
    "sphinx.ext.viewcode",          # add [source] links to API pages
    "sphinx.ext.intersphinx",       # cross-link to NumPy / Python docs
    "sphinx.ext.autosummary",       # generate summary tables
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# autodoc options
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # render type hints in Parameters section
add_module_names = False

# ---------------------------------------------------------------------------
# napoleon (NumPy docstring parser)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "sustainable-jsp"

# GitHub Pages project site: https://<user>.github.io/<repo>/
# Set DOCS_BASE_URL=/sustainable-jsp/ in CI; leave unset for local builds.
html_base_url = os.environ.get("DOCS_BASE_URL", "")
