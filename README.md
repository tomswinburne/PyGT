# Python codes for KTN analysis - GT + sensitivity
>Tom Swinburne, CNRS, 2019
>swinburne@cinam.univ-mrs.fr

## packages
1. It is nice to install the [`tqdm`](https://pypi.org/project/tqdm/) package to having loading bars, using e.g.
```
pip install tqdm
```

## files
1. `lib/*.py` i/o tools, matrix manipulations, GT renormalizations
2. `just_gt.py` numerical tests of observable extraction with varying degrees of path-based renormalization. All provably identical up to floating point error- therefore tests floating point error
3. `no_sample_test.py` fake sampling to test for convergence
4. `plot.py` run after `no_sample_test.py` to see results!
