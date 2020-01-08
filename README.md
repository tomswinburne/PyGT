# Python codes for KTN analysis - GT + sensitivity
>Tom Swinburne, CNRS, 2019
>swinburne@cinam.univ-mrs.fr

## Packages / Requirements
1. You need Python 3.* . Check by running
```
`python --version
```
2. Also need `numpy`, `scipy` and `matplotlib`, and it is nice to have the [`tqdm`](https://pypi.org/project/tqdm/) package to have loading bars. These can all be installed with
```
pip install -r requirements.txt
```
with the included `requirements.txt` file

## Code
1. `lib/*.py` i/o tools, matrix manipulations, GT renormalizations
2. `just_gt_scripts/*.py` numerical tests of observable extraction with varying degrees of path-based renormalization. All provably identical up to floating point error- therefore tests floating point error
3. `fake_sample_scripts/*.py` fake sampling to tests for convergence. No path GT yet, to keep things simple!
4. `gt_rate_calc.py` and `graph_transformation_notebook.ipynb` look at rate calculations following various GTs this is where numerical issues are particularly notable!
