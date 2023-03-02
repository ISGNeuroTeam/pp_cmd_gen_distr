import numpy as np
import pandas as pd


def _pert(a, b, c, size=1, lamb=4):
    if b <= a:
        raise ValueError('Wrong params for PERT distribution: must be b > a')
    if c <= b:
        raise ValueError('Wrong params for PERT distribution: must be c > b')

    r = c - a
    alpha = 1 + lamb * (b - a) / r
    beta = 1 + lamb * (c - b) / r
    return a + np.random.beta(alpha, beta, size=size) * r


def _bernulli(p, size=1):
    if p < 0 or p > 1:
        raise ValueError(f'Wrong parameter for Bernulli distribution: must be 0 <= a <= 1, having a={p}')

    return np.random.binomial(1, p, size=size)


def _binominal(n, p, size=1):
    if p < 0 or p > 1:
        raise ValueError(f'Wrong parameter for Binominal distribution: must be 0 <= p <= 1, having {p=}')
    if n < 0:
        raise ValueError(f'Wrong parameter for Binominal distribution: must be n >= 0, having {n=}')

    return np.random.binomial(n, p, size=size)


def _normal(mean, deviation, size=1):
    if deviation < 0:
        raise ValueError(f'Wrong parameter for Normal distribution: must be deviation >= 0, having {deviation=}')

    return np.random.normal(loc=mean, scale=deviation, size=size)


def _poisson(lam, size=1):
    if lam < 0:
        raise ValueError(f'Wrong parameter for Poisson distribution: must be lam >= 0, having {lam=}')

    return np.random.poisson(lam=lam, size=size)


def _triangular(left, mode, right, size=1):
    if left > mode:
        raise ValueError(
            f'Wrong parameter for Triangular distribution: must be min <= mode, having {left=} and {mode=}')
    if mode > right:
        raise ValueError(
            f'Wrong parameter for Triangular distribution: must be mode <= max, having {mode=} and {right=}')

    return np.random.triangular(left=left, mode=mode, right=right, size=size)


def _uniform(low, high, size=1):
    if low > high:
        raise ValueError(
            f'Wrong parameter for Uniform distribution: must be low < high, having {low=} and {high=}')
    
    return np.random.uniform(low=low, high=high, size=size)


DISTRIBUTIONS = {
    'pert': _pert,
    'bernulli': _bernulli,
    'binominal': _binominal,
    'normal': _normal,
    'poisson': _poisson,
    'triangular': _triangular,
    'uniform': _uniform
}


def generate(name, size, **params):
    fn = DISTRIBUTIONS[name]
    return pd.DataFrame(fn(size=size, **params), columns=['value'])
