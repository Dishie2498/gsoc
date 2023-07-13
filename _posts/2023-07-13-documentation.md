---
title: Documentaion in Sphinx
layout: post
post-image: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQNsTNfNOigQe2bk5Lw_UeXCuq9UdxEAL0zRg&usqp=CAU"
description: A short tutorial on how to build documentation in Sphinx
tags:
- documentation
- sphinx
- autosummary
---
# Steps on generating your own documentation using Sphinx
### Initial set-up
* Create a folder 'docs' to contain your documentation inside your repo
```
cd docs
sphinx-quickstart
```
Enter 'y' for all the following questions.

* The above command would have generated files and folders inside the docs directory. <br>
1. Inside the `conf.py` file, uncomment lines #13, #14, #15
2. In line #15, change the absolute path to ".."

* In conf.py, add the extensions you require in the `extensions` list which would initially be empty. <br>
We have used the following to build the entire documentation: <br>
```extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    "sphinx.ext.extlinks",
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex',
    'sphinx_panels',
    'numpydoc',
    'sphinx_copybutton'
]```

* Choose the theme of your choice<br>
Inside conf.py, you can set the `html_theme` to any of your choice from the [Sphinx themes](https://sphinx-themes.org/). <br>
We have used `html_theme = 'pydata_sphinx_theme'`.

### To generate autosummary
* Inside `index.rst`, add the 'modules' to the toctree in the given manner.
```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```
* Outside the docs directory, run the following command
```
cd .. (to go out of docs)
sphinx-apidoc -o docs/ <your current directory name>
```
A list of .rst files will be generated inside docs.

### Building html pages
* Inside the docs directory, run `make html`
```
cd docs
make html
```
* Inside docs/_build/html, there will be many html files. <br>
You can host `index.html` locally to see your documentation on localhost.

## Tutorial Video
<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/11kjhMHtKuwCT8kFiw5ttyjig2HDlOefd/view?usp=sharing" frameborder="0" allowfullscreen="true"> </iframe>
</figure>
