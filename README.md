
# Multiscale EM algorithm

This repository contains the code for the multiscale expectation-maximization (EM) algorithm to extract latent states (and learn the multiscale state-space model) from the combined spiking activity (in form of binary point process Poisson observations) and field activity (in form of continuous linear Gaussian observations, e.g., local field potentials or LFP). 

This algorithm is used for the paper "[Multiscale low dimensional motor cortical state dynamics predict naturalistic reach-and-grasp behavior](https://www.nature.com/articles/s41467-020-20197-x)". The mathematical derivation of the algorithm could be found [here](https://ieeexplore.ieee.org/abstract/document/8698887).

## Installation guide

To use the software, you just need to download this repository or clone the repository using git. Make sure to include all the repository folders to the path. For a normal desktop computer that already has MATLAB installed on it, this process takes less than 5 mins. Installing MATLAB usually takes about a couple hours.

## Dependencies

This repository does not need any extra dependencies apart from the built-in MATLAB functions.

## User guide

The main function is located at [EM_multiscale_unsupervised_function.m](https://github.com/SalarAbb/emrelease/blob/main/functions_main/EM_multiscale_unsupervised_function.m). To get familiar with how the algorithm works, there exists a demo script [demo.m](https://github.com/SalarAbb/emrelease/blob/main/example/demo.m) which generates a simulated time-series of multiscale spike-LFP activity from a state-space model, learns it with the multiscale EM algorithm and visualizes the learned eigenvalues compared to the true systems eigenvalues. To run on your data, you just need to prepare the lfp features time-series (Y) and spike time-series (N), and set some settings parameters of your interest. The detailed instructions could be found in the demo script and also inside functions.

## Version compatibility

The code has been tested on the following MATLAB and operating system versions.
- MATLAB 2017a, Windows 10 Pro (version 1909)
- MATLAB 2019b, Windows 10 Pro (version 1909)
- MATLAB 2018b, CentOS Linux (version 7)

## License
Copyright (c) 2020 University of Southern California  
See full notice in [LICENSE.md](LICENSE.md)  
Hamidreza Abbaspourazad and Maryam M. Shanechi  
Shanechi Lab, University of Southern California
