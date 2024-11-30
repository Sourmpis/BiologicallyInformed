# Biologically Informed code

This is the code for the paper:
C. Sourmpis, C. Petersen, W. Gerstner & G. Bellec
[*Biologically informed cortical models predict optogenetic perturbations*](https://www.biorxiv.org/content/10.1101/2024.09.27.615361v1.full).

Contact:
[christos.sourmpis@epfl.ch](mailto:christos.sourmpis@epfl.ch)


## Glossary
1) [Installation](#Installation)
2) [Download recorded data from Esmaeli et al. 2021](#download-recorded-data)
3) [Generate synthetic data](#generate-synthetic-data)
4) [Generate paper figures](#generate-figures-from-a-pre-trained-model)

## Installation
We suggest running the code using docker and in particular the image nvcr.io/nvidia/pytorch:23.05-py3.
Once your container is running you can install the code by running the following command:

```bash
pip install -e .
```
Now you should be able to run the code.

## Download recorded data
Be aware that the full data is ~55GB, but we will use only ~3GB in the end.

In order to use the recorded data you can do it manually 
1. download the data from [here](https://zenodo.org/record/4720013), 
2. unzip 
3. from the Electrophysiology folder keep the spike_data_v9.mat 
4. put the spike_data_v9.mat in the datasets folder 

or run the following commands:

```bash
wget https://zenodo.org/record/4720013/files/Esmaeili_data_code.zip -P datasets
unzip datasets/Esmaeili_data_code.zip -d tmp
mv tmp/Electrophysiology/Data/spikeData_v9.mat datasets/spikeData_v9.mat
rm -r tmp
```

Now we need to prepare the data. In order to do so run the following command:

```bash
python datasets/datastructure2datasetandvideo_Vahid.py
```

## Generate synthetic data
For the synthetic data just run the commands:

```bash
python datasets/pseudodata2areas.py
python infopath/utils/parameter_recovery.py
```
This will generate the data needed for Figure 1-2 and the data for some of the supplementary Figures.


## Generate figures from a pre-trained model

The code is sufficient in order to generate all the figures of the paper, in the folder `Figures` one can find the notebooks to generate all the panels.
