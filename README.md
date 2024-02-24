# GravitationalWaveDetection

This repository contains code and resources for a project focused on detecting gravitational wave signals from the mergers of binary black holes using machine learning techniques.

## Project Overview

The detection of gravitational waves (GW) in 2015 marked a monumental breakthrough in astrophysics, enabling the observation of phenomena such as binary black hole mergers and neutron star collisions. This project aims to develop machine learning models for the analysis of simulated GW time-series data obtained from Earth-based detectors.

## Objectives

The primary objectives of this project include:

- Building machine learning models for the identification of GW signals amidst detector noise.
- Leveraging data science techniques to handle complex and massive GW datasets efficiently.
- Exploring algorithms for noise removal, data conditioning, and signal characterization.

### Data Details

- The dataset contains simulated gravitational wave measurements.
- Each data sample (npy file) contains 3 time series (1 for each detector), each spanning 2 seconds and sampled at 2,048 Hz.
- The integrated signal-to-noise ratio (SNR) is classically the most informative measure of how detectable a signal is. A typical level of detectability is when this integrated SNR exceeds ~8.
- Parameters determining the waveform's exact form (e.g., masses, sky location, distance, black hole spins, etc.) have been randomized according to astrophysically motivated prior distributions but are not provided as part of the competition data.

## Files Included

- `train/`: Contains the training set files, with one npy file per observation. Labels are provided in the `training_labels.csv` file.
- `test/`: Contains the test set files. The objective is to predict the probability that the observation contains a gravitational wave.

## Project Structure

The repository structure is organized as follows:

- `src/`: Contains the source code.
  - `data/`: Data processing utilities.
  - `models/`: Implementation of machine learning models.
  - `utils/`: Utility functions and visualization tools.

## Visualizations and Model Summary

The `images` contains visualizations and model summaries related to the project. 

## Model Training

The model was trained for 2 epochs using the training dataset. Training for a longer duration might improve performance further.

## Model Performance

The trained model achieved the following performance on the validation set:

- **Training Loss:** 0.4589
- **Training AUC:** 0.8404
- **Validation Loss:** 0.4552
- **Validation AUC:** 0.8437

These metrics indicate the model's ability to generalize well to unseen data, with a competitive AUC score suggesting strong predictive capability.

## Acknowledgments

This project is part of the G2Net collaboration, which aims to bring together expertise in gravitational wave physics, geophysics, computing science, and robotics for advancing GW research.



