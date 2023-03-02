import pandas as pd
from otlang.sdk.syntax import Keyword, Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax
from ts_forecasting import distr

PARAMETERS = {
    'pert': ['a', 'b', 'c'],
    'bernulli': ['p'],
    'binominal': ['n', 'p'],
    'normal': ['mean', 'deviation'],
    'poisson': ['lam'],
    'triangular': ['left', 'mode', 'right'],
    'uniform': ['low', 'high'],
    'weibull': ['form', 'scale'],
    'chisquare': ['df'],
    'beta': ['alpha1', 'alpha2'],
    'pareto': ['theta', 'a'],
    'discrete': ['values', 'probabilities']
}


class GenDistrCommand(BaseCommand):
    # define syntax of your command here
    syntax = Syntax(
        [
            Positional("distr_name", required=True, otl_type=OTLType.STRING),
            Keyword("size", required=False, otl_type=OTLType.INTEGER),
            Keyword("to_file", required=False, otl_type=OTLType.TEXT),

            # PERT params:
            Keyword("a", required=False, otl_type=OTLType.NUMERIC),
            Keyword("b", required=False, otl_type=OTLType.NUMERIC),
            Keyword("c", required=False, otl_type=OTLType.NUMERIC),

            # Bernulli parameter
            Keyword("p", required=False, otl_type=OTLType.NUMERIC),

            # Binominal parameter
            Keyword("n", required=False, otl_type=OTLType.NUMERIC),

            # Normal distribution parameters
            Keyword("mean", required=False, otl_type=OTLType.NUMERIC),
            Keyword("deviation", required=False, otl_type=OTLType.NUMERIC),

            # Poisson parameter
            Keyword("lam", required=False, otl_type=OTLType.NUMERIC),

            # Triangular parameter
            Keyword("left", required=False, otl_type=OTLType.NUMERIC),
            Keyword("mode", required=False, otl_type=OTLType.NUMERIC),
            Keyword("right", required=False, otl_type=OTLType.NUMERIC),

            # Uniform parameters
            Keyword("low", required=False, otl_type=OTLType.NUMERIC),
            Keyword("high", required=False, otl_type=OTLType.NUMERIC),

            # Weibull parameters
            Keyword("form", required=False, otl_type=OTLType.NUMERIC),
            Keyword("scale", required=False, otl_type=OTLType.NUMERIC),

            # Chi-Square parameter
            Keyword("df", required=False, otl_type=OTLType.INTEGER),

            # Beta parameters
            Keyword("alpha1", required=False, otl_type=OTLType.NUMERIC),
            Keyword("alpha2", required=False, otl_type=OTLType.NUMERIC),

            # Pareto parameters
            Keyword("theta", required=False, otl_type=OTLType.NUMERIC),

            # Discrete paramemeter
            Keyword("values", required=False, otl_type=OTLType.TEXT),
            Keyword("probabilities", required=False, otl_type=OTLType.TEXT),
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress('Start gen_distr command')

        # Check distribution name
        distr_name = self.get_arg('distr_name').value
        if distr_name not in distr.DISTRIBUTIONS:
            raise ValueError(
                f'Unsupported distribution. Known distribution are: {", ".join(distr.DISTRIBUTIONS.keys())}')

        # Check size
        size = self.get_arg('size').value or 1

        # Output file
        to_file = self.get_arg('to_file').value

        # Check params
        params = dict()
        for param_name in PARAMETERS[distr_name]:
            if (param_val := self.get_arg(param_name).value) is None:
                raise ValueError(f'Missing param for {distr_name} distribution: {param_name}')
            params[param_name] = param_val

        result_df = distr.generate(name=distr_name, size=size, **params)

        if to_file:
            result_df.to_parquet(to_file)

        return result_df
