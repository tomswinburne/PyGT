# PyGT : Graph transformation and reduction in Python

> Tom Swinburne, CNRS-CINaM, swinburne at cinam.univ-mrs.fr

> Deepti Kannan, U Cambridge

v0.3.0 :copyright: TD Swinburne and D Kannan 2020

## Quick installation
```pip install PyGT```

If this looks unfamiliar, please see below


Beta version of code used in the following papers :
T.D. Swinburne and D.J. Wales, *Defining, Calculating, and Converging Observables of a Kinetic Transition Network*, J. Chemical Theory and Computation (2020), [link](https://doi.org/10.1021/acs.jctc.9b01211)

T.D. Swinburne, D. Kannan, D.J. Sharpe and D.J. Wales, *Rare Events and First Passage Time Statistics From the Energy Landscape*,
Submitted to J. Chemical Physics (2020)

More functionality will be added soon.

## Documentation and online examples

 - Example notebooks can be run online with binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tomswinburne/PyGT/master?filepath=examples%2Fbasic_functions.ipynb)

 - Please see full documentation at [readthedocs](https://pygt.readthedocs.io)



## Recommended installation with pip
- We recommend using a `virtualenv` with e.g. [conda](https://docs.conda.io/en/latest/miniconda.html)-
```bash
	conda create --name PyGTenv python=3.5
	conda activate PyGTenv
```
- One can then safely install PyGT using `pip` with
```bash
	pip install PyGT
```

## Run the examples locally
- Install [jupyter](https://jupyter.org/install) notebook if required, inside the same `virtualenv`
```
	conda install -c conda-forge notebook
```
- Open the example notebook
```
  cd examples
  jupyter-notebook basic_functions.ipynb
```
