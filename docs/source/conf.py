project = 'Piscis'
copyright = '2023, William Niu'
author = 'William Niu'
release = '0.0.1'

extensions = [
    'sphinx.ext.intersphinx',
    'autoapi.extension',
    'numpydoc'
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'flax': ('https://flax.readthedocs.io/en/latest/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
}
autoapi_dirs = ['../../piscis']

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
