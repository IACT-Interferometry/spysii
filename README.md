Project description
==============

spysii is a software in python for stellar intensity interferometry


Documentation
-------------

There is not yet public documentation. Simply follow the notebooks within the notebooks folder.

Dependencies
------------

Current dependencies (listed within the environment.yml):

```python
  - python=3.9
  - numpy=1.23.0
  - astropy=5.0.4
  - pip
  - bokeh~=2.4.2
  - click
  - coloredlogs
  - scipy
  - matplotlib
  - lmfit
  - yaml
  - uproot
  - scipy
  - jupyter
  - astroquery
  - tqdm
  - corner
  - pandas
  # Needed for PMOIRED
  - ipympl
  # Needed for documentation with markdown format
  - sphinx
  - sphinx-automodapi
  - pydata-sphinx-theme
  - sphinx-click
  - nbsphinx
  - sphinx-design
```

You may create a conda environment with these libraries (that are probably overkill) with the following command:
```bash
conda env create --name spysii --file=environments.yml
```
If you do, make sure every time you want to use this software, you load the environment:
```bash
conda activate spysii
```
Note we **strongly recommend using mamba** (and not conda).

Installation
--------------

The package can be installed by downloading it:

```shell
git clone git@github.com:TarekHC/spysii.git
```

And then just installing it with pip from the parent folder:
```shell
cd spysii
pip install .
```
