# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'esgaliant'
author = 'Patrick CN Martin'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate from docstrings
    'sphinx.ext.napoleon',     # Support Google/NumPy style
    'sphinx.ext.viewcode',     # Add source code links
    'sphinx.ext.intersphinx',  # Link to other docs
    'sphinx_autodoc_typehints', # Better type hint display
]

# Theme
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme

# Napoleon settings for Google/NumPy style
napoleon_google_docstring = True
napoleon_numpy_docstring = True