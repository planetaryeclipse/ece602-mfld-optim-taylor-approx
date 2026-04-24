# ece602-mfld-optim-taylor-approx

This is the repository containing all accompanying code to the project report titled "Manifold Optimization via Approximated Geodesic Retractions for Safety Guarantees in Embedded Geometric Control Systems" produced as part of the course project deliverables for ECE 602 Introduction to Optimization.

Note that this repo is intended to be run from PyCharm and is configured with an `.idea` subdirectory You can likely run all the scripts/notbooks in this repository without much issue otherwise but you will need to ensure that you have added `.src` to your `PYTHONPATH` due to the presence of custom libraries needed for manifold optimization.

This project is structured according to the following:
- `data` contains the various subdirectories used to store the results obtained from running each test optimization process
- `derivations` contains a notebook that can be utilized to generate higher-order Taylor series terms of a geodesic vector field. These are the $f^{(k)}$ values using the notation provided in the report.
- `src` main source directory containing the custom libraries for manifold optimization, data generation, and analysis
  - `diff_mfld` custom differential geometry library based on the first draft I developed for use in differentiable manifold optimization layers (original version was taken from here https://github.com/planetaryeclipse/research-geo-diff-opt-layer)
    - `geodesic` implements the geodesic functions (Riemannian exponential and logarithmic maps)
    - `geometry` implements connections, manifold functions ($C^\infty(\mathcal{M})$), and the metric field construct (effectively the implementation of a section of $T^{(0,2)}\mathcal{M}$)
  - `optim` custom manifold optimization library implementing the various algorithms described in the associated report. Note that the subdirectories here are failry self-explanatory but there are Python files in the root folder that provide access to each method via an enum
  - `testing` contains the utility file `testing_metrics.py` employeed by the test data generation notebooks. This file contains the `euclid`, `scaled`, `coupled`, and `asymmetric` Riemannian metrics (with scaling) as described in the associated paper
  - `analysis` contains the notebooks, utility Python files, and subdirectories to perform analysis on the resulting test files and store the resulting generated figures

> Note that the main differences in custom library implementation comes down to the separation of the geometry from optimization algorithms as compared to the preceding version. Further, there are more optimization algorithms available in this repository.

> Further note that the structure of the optimization library has been left fairly similar in order to be integrated back into the other repository to be used with the differentiable manifold optimization layers.

Note that to generate/run the optimization algorithms, use one of the notebooks located in the `src` directory. Ensure that your desired configuration is correct, i.e. ensure that desired scaling, metrics, approximation methods, etc., are uncommented if desired. From there you can run the algorithms. Note that in the main loop iterating over the various configurations, the result of each solved optimization algorithm *may* be commented. This was unfortunately due to excessive memory consumption if either the Euclidean fallback message was triggered or if the displayed result was sufficiently large enough. This also occurs when multiple notebooks are running at the same time. To show the result, and position history, please simply uncomment this line.