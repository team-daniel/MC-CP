# MC-CP
This repository contains code from my paper on Robust Uncertainty Quantification using Conformalised Monte Carlo Prediction.

### Introduction
Deploying deep learning models in safety-critical applications remains a very challenging task, mandating the provision of assurances for the dependable operation of these models. Uncertainty quantification (UQ) methods estimate the model’s confidence per prediction, informing decision-making by considering the effect of randomness and model misspecification. Despite the advances of state-of-the-art UQ methods, they are computationally expensive or produce conservative prediction sets/intervals. We introduce MC-CP, a novel hybrid UQ method that combines a new adaptive Monte Carlo (MC) dropout method with conformal prediction (CP). MC-CP adaptively modulates the traditional MC dropout at runtime to save memory and computation resources, enabling predictions to be consumed by CP, yielding robust prediction sets/intervals. Throughout comprehensive experiments, we show that MC-CP delivers significant improvements over advanced UQ methods, like MC dropout, RAPS and CQR, both in classification and regression benchmarks. MC-CP can be easily added to existing models, making its deployment simple.

### Adaptive Monte-Carlo Dropout
Our proposed Adaptive MC Dropout method is a novel MC Dropout method that can save computational resources compared to the original method. The motivation underpinning adaptive MC dropout originates from the observation that each forward pass corresponds to a particular DL model instantiation that adds unique variance to the prediction distribution. Some of these DL model instantiations, informed by MC dropout forward passes, can produce similar or even the exact same prediction. Hence, although the prediction variance might be large initially, as the number of forward passes increases, the variance value becomes smaller, indicating that the inference process has converged.

Adaptive MC Dropout requires a few hyperparameters: the model (f), the input/data (x), the maximum number of forward passes (K), the threshold (delta), and the patience (P).

Given a new input (x), the method performs up to (K) forward passes over the model (f) to produce the predictive posterior mean as the final prediction and the variance of the predictive posterior as the prediction uncertainty. Unlike conventional MC dropout, our algorithm uses the hyperparameters threshold (delta) and patience (P) to detect the convergence and terminate early. The threshold parameter (delta) denotes the maximum difference in variance required to trigger that the class/quantile prediction has likely converged. Patience (P) signifies the number of consecutive forward passes where all classes/quantiles are below (delta) to stop the execution early.

### Conformalised Monte-Carlo Prediction

### Getting Started

### Files
This repository contains the following files:
* `AAAISupplementaryCode.ipynb` - A detailed jupyter notebook that will guide readers through our proposed methods.
* `Concrete_Data.csv` - Supplementary data for regression.
* `abalone.data.csv` - Supplementary data for regression.
* `CASP.csv` - Supplementary data for regression.

### Publicly Available Datasets

