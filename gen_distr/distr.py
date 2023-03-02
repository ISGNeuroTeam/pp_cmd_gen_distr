import numpy as np
from numpy.random import default_rng
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


def _weibull(form, scale, size=1):
    rng = default_rng()

    return scale * rng.weibull(a=form, size=size)


def _chisquare(df, size=10):
    if df < 1:
        raise ValueError(
            f'Wrong parameter for Chi-Square distribution: must be df > 0, having {df=}')

    rng = default_rng()
    return rng.chisquare(df=df, size=size)


def _beta(alpha1, alpha2, size=1):
    if alpha1 < 1:
        raise ValueError(
            f'Wrong parameter for Beta distribution: must be alpha1 > 0, having {alpha1=}')
    if alpha2 < 1:
        raise ValueError(
            f'Wrong parameter for Beta distribution: must be alpha2 > 0, having {alpha2=}')

    rng = default_rng()
    return rng.beta(alpha1, alpha2, size=size)


def _pareto(theta, a, size=1):
    if theta < 1:
        raise ValueError(
            f'Wrong parameter for Pareto distribution: must be theta > 0, having {theta=}')
    if a < 1:
        raise ValueError(
            f'Wrong parameter for Pareto distribution: must be a > 0, having {a=}')
    rng = default_rng()
    return (rng.pareto(theta, size=size) + 1) * a


def _discrete(values, probabilities, size=1):
    values_list = [float(x) for x in values.split(';') if len(x) > 0]
    probabilities_list = [float(x) for x in probabilities.split(';') if len(x) > 0]

    if len(values_list) == 0:
        raise ValueError(f'Zero value elements found. Must be at least one')
    if len(probabilities_list) == 0:
        raise ValueError(f'Zero probability elements found. Must be at least one')
    if len(values_list) != len(probabilities_list):
        raise ValueError(f'values and probabilities must have same size')

    rng = default_rng()
    return rng.choice(a=values_list, size=size, p=probabilities_list)


DISTRIBUTIONS = {
    'pert': _pert,
    'bernulli': _bernulli,
    'binominal': _binominal,
    'normal': _normal,
    'poisson': _poisson,
    'triangular': _triangular,
    'uniform': _uniform,
    'weibull': _weibull,
    'chisquare': _chisquare,
    'beta': _beta,
    'pareto': _pareto,
    'discrete': _discrete
}


def generate(name, size, **params):
    fn = DISTRIBUTIONS[name]
    return pd.DataFrame(fn(size=size, **params), columns=['value'])
