# Python codes for KTN analysis - GT + sensitivity
>Tom Swinburne, CNRS, 2019
>swinburne@cinam.univ-mrs.fr

## packages
1. It is nice to install the [`tqdm`](https://pypi.org/project/tqdm/) package to have loading bars, using e.g.
```
pip install tqdm
```

## files
1. `lib/*.py` i/o tools, matrix manipulations, GT renormalizations
2. `just_gt.py` numerical tests of observable extraction with varying degrees of path-based renormalization. All provably identical up to floating point error- therefore tests floating point error
3. `fake_sample_no_path_gt.py` fake sampling to test for convergence. No path GT yet, to keep things simple!
4. `plot.py` run after `no_sample_test.py` to see results!
