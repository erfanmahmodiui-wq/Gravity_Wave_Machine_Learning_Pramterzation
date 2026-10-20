# Gravity Wave Analysis and Machine Learning Parameterization

This repository contains the code used for extracting gravity-wave (GW) signals, computing wave-action density spectra, and developing and evaluating an interpretable deep learning-based gravity-wave parameterization for the lower stratosphere.

The workflow consists of two main components:

1. **Gravity-wave analysis:** Extraction of gravity-wave signals from resolved atmospheric fields, computation of wave-action density spectra, and coarse-graining onto a unified three-dimensional wavenumber grid.
2. **Machine learning parameterization:** Dataset preprocessing, convolutional neural network (CNN) training, model evaluation, and explainable artificial intelligence (XAI) analysis.

## 1. Gravity-Wave Analysis

**Directory:** `gravity_wave_analysis/`

This component extracts gravity-wave signals and computes the corresponding wave-action density spectra.

The processing pipeline consists of the following steps:

1. Read the two-dimensional spatial fields at consecutive time steps.
2. Apply a temporal Fast Fourier Transform (FFT) to construct a three-dimensional spectrum.
3. Decompose the spectrum to isolate gravity-wave signals using the gravity-wave dispersion relation.
4. Compute the gravity-wave action density.
5. Coarse-grain the resulting spectra onto a unified three-dimensional wavenumber grid \((k,l,m)\).

### Main scripts

**`gravity_wave_analysis/run_job.sh`**

SLURM submission script controlling the time windows used for spectral analysis.

- The `#SBATCH --array` parameter specifies the time-step indices to process, ranging from 0 to 72.
- Each array task processes 26 hours of data using a 24-hour analysis window.
- The analysis window is shifted by one hour to augment the dataset, producing two temporal samples for each coarse-grid cell.

**`gravity_wave_analysis/avrage_slide.py`**

Dynamically manages the different processing steps of the gravity-wave analysis pipeline.

**`gravity_wave_analysis/timespectrum.py`**

Computes the three-dimensional Fourier spectrum by applying the FFT along the temporal dimension of the two-dimensional spatial fields.

**`gravity_wave_analysis/decomposition.py`**

Separates gravity-wave signals from the three-dimensional spectrum using the gravity-wave dispersion relation.

**`gravity_wave_analysis/wave_action.py`**

Computes gravity-wave action density and coarse-grains the resulting spectra onto a common three-dimensional \((k,l,m)\) wavenumber grid.

## 2. Gravity-Wave Machine Learning Parameterization

**Directory:** `Gravity_Wave_Machine_Learning_Pramterzation/`

This component contains the preprocessing, training, evaluation, and interpretation of the deep learning-based gravity-wave parameterization.

### 2.1 Data preprocessing

**`preprocessing/build_dataset.py`**

Constructs the machine learning datasets for each month by combining the extracted gravity-wave action density spectra with the corresponding resolved atmospheric flow variables.

The samples are separated into land and ocean groups and organized into training, validation, and testing datasets.

**`preprocessing/green_gauss.py`**

Computes the horizontal gradients of the resolved atmospheric flow using the Green–Gauss method.

### 2.2 Model training

**`model/model.py`**

Loads and preprocesses the monthly datasets, concatenates the selected months, and trains the convolutional neural network (CNN) to predict gravity-wave action density spectra.

**Note on training epochs:** The number of epochs is set to 50 in the script due to computational time limitations imposed by the cluster. Training was continued through repeated runs until the stopping criterion described in the accompanying manuscript was satisfied.

### 2.3 Model evaluation

**`evaluation/metrics/`**

Contains scripts for evaluating model performance, including:

- Statistical metrics in logarithmic space.
- Statistical metrics in the original physical space.
- Evaluation of relevant physical quantities derived from the predicted gravity-wave spectra.

The computed metrics and associated results are saved for subsequent analysis.

**`evaluation/xai/`**

Contains scripts for computing Shapley values to interpret the trained model and quantify the contributions of input features to its predictions.

Additional scripts are provided for calculating physical quantities, including gravity-wave momentum fluxes, and for generating diagnostic plots.

## 3. Notebooks and Visualization

**`notebooks/paper_vis.ipynb`**

Generates the visualizations used in the accompanying manuscript.

**`notebooks/out_sample_testing.ipynb`**

Evaluates model generalization using independent out-of-month and out-of-year samples.

Other notebooks and scripts support additional analyses and visualizations, including power-law relationships and gravity-wave spectral characteristics.

## 4. Dataset

The corresponding dataset is archived separately on Zenodo and contains land and ocean samples for January, August, and November.

- **January and August:** Used for model development, including training, validation, and testing.
- **November:** Used for independent out-of-month and out-of-year evaluation.

The target variables are gravity-wave action density spectra represented on a unified \(6 \times 6 \times 4\) wavenumber grid.

**Dataset DOI:** https://doi.org/10.5281/zenodo.23493288

## 5. Reference

This repository accompanies the manuscript:

*Interpretable Deep Learning-Based Modeling of Gravity Waves in the Lower Stratosphere.*

## 6. Citation

If you use this code or the associated dataset, please cite the accompanying manuscript and the corresponding Zenodo records.

**Software DOI:** [Add Zenodo software DOI]

**Dataset DOI:** https://doi.org/10.5281/zenodo.23493288