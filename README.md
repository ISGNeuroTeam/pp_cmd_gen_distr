# pp_cmd_gen_distr
Postprocessing command "gen_distr"

Usage example

For PERT distribution:
`... | gen_distr 'pert' min=10 moda=50 max=110 size=20`

For Bernulli distribution:
`... | gen_distr 'bernulli' p=1.4 size=20` (0 <= p <= 1)

For Binominal distribution:
`... | gen_distr 'binominal' n=20 p=0.4 size=20` (0 <= p <= 1)

For Normal distribution:
`... | gen_distr 'normal' mean=20 deviation=0.4 size=20` (deviation > 0)

For Poisson distribution:
`... | gen_distr 'poisson' lam=20 size=20` (lam > 0)

For Triangular disctribution:
`... | gen_distr 'triangular' left=10 mode=50 right=110 size=20` (left <= mode <= right)

For Uniform (_равномерное_) distribution:
`... | gen_distr 'uniform' low=10 high=110 size=20` (low <= high)

For Weibull distribution:
`... | gen_distr 'weibull' form=10 scale=110 size=20`

For Chi-Square distribution:
`... | gen_distr 'chisquare' df=10 size=20` (df > 0)

For Beta distribution:
`... | gen_distr 'beta' alpha1=10 alpha2=15 size=20` (alpha1 > 0, alpha2 > 0)

For Pareto distribution:
`... | gen_distr 'pareto' theta=10 a=15 size=20` (theta > 0, a > 0)

For Discrete distribution:
`... | gen_distr 'discrete' values='1;2;3;4;5' probabilities='0.2;0.2;0.2;0.2;0.2' size=20` (amount of values and probabilities should be the same. All probabilities must sum up to 1.)

## Getting started
###  Prerequisites
1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installing
1. Create virtual environment with post-processing sdk 
```bash
make dev
```
That command  
- creates python virtual environment with [postprocessing_sdk](https://github.com/ISGNeuroTeam/postprocessing_sdk)
- creates `pp_cmd` directory with links to available post-processing commands
- creates `otl_v1_config.ini` with otl platform address configuration

2. Configure connection to platform in `otl_v1_config.ini`

### Test gen_distr
Use `pp` to test gen_distr command:  
```bash
pp
Storage directory is /tmp/pp_cmd_test/storage
Commmands directory is /tmp/pp_cmd_test/pp_cmd
query: | otl_v1 <# makeresults count=100 #> |  gen_distr 
```
