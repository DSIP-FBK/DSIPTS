import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "DSIPTS"
author = "Andrea Gobbi (agobbifbk.eu)"
release = "1.1.44"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
]

autosummary_generate = True         # generate stub files automatically
autoclass_content = "init"         # use __init__ docstring as class doc

autodoc_default_options = {
    "undoc-members": False,         # skip undocumented methods
    "special-members": "",          # do NOT include __init__ separately
}

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
