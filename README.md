falcon
======

For more information:

* [Official code website](https://github.com/bittremieux/falcon)

The _falcon_ spectrum clustering tool uses advanced algorithmic techniques for highly efficient processing of millions of MS/MS spectra. First, high-resolution spectra are binned and converted to low-dimensional vectors using feature hashing. Next, the spectrum vectors are used to construct nearest neighbor indexes for fast similarity searching. The nearest neighbor indexes are used to efficiently compute a sparse pairwise distance matrix without having to exhaustively compare all spectra to each other. Finally, density-based clustering is performed to group similar spectra into clusters.

The software is available as open-source under the BSD license.

If you use _falcon_ in your work, please cite the following publication:

- Wout Bittremieux, Kris Laukens, William Stafford Noble, Pieter C. Dorrestein. **Large-scale tandem mass spectrum clustering using fast nearest neighbor searching.** _publication pending_ (2021).

Running falcon
--------------

To run _falcon_:

1. Configure the settings in `config.py`.
2. Run _falcon_ using: `python falcon.py`.

_falcon_ was developed using Python 3.8. See [environment.yml](https://github.com/bittremieux/falcon/blob/main/environment.yml) for a list of dependencies. A suitable conda environment can easily be created using `conda env create -f environment.yml`.

Contact
-------

For more information you can visit the [official code website](https://github.com/bittremieux/falcon) or send an email to <wbittremieux@health.ucsd.edu>.
