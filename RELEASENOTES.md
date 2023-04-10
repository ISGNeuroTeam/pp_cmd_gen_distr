## [0.3.1] - added functionality to discrete distribution
use `use_dataframe=yes` to read data from dataframe, read with readFile function.

## [0.2.1] - fixed PARETO distribution lacking input parameter
`a` input numeric parameter was lacking due to previous renaming of the PERT parameters  

## [0.2.0] - changed PERT parameter names to its origin
`a` to `min`, `b` to `moda`, c to `max`

## [0.1.0] - All 12 required distributions are now added to one postprocessing command:
- pert
- bernulli
- binominal
- normal
- poisson
- triangular
- uniform
- weibull
- chisquare
- beta
- pareto
- discrete