# Weather and Urban Centre Emissions 

## Overview
This project investigates the MET Office subset of the WeatherBench dataset that contains information on climate factors (precipitation, u-wind, v-wind and temperature) for 12 locations worldwide. 

## AI Declaration
I acknowledge the use of ChatGPT (https://chat.openai.com/) to assist with code debugging. The output's from the prompts used were then analysed carefully and correct lines of code were modified in the notebook. This reduced the amount of time needed for debugging and allowed for more focus on data analysis and interpretation. 

## Datasets 
### Initial Datasets
For this research, multiple datasets have been combined to investigate links between climate, socioeconomic factors and natural disasters. Below are links to the datasets used for this analysis: 
* [Applied Data Science - Met Office Dataset](https://zenodo.org/records/10624903)
* [Emergency Events Database(EM-DAT)](https://public.emdat.be/)
* [Geocoded Disasters (GDIS) Dataset](https://data.nasa.gov/dataset/geocoded-disasters-gdis-dataset)
* [Global Human Settlement Layer - Urban Centre Database (GHSL UCDB)](https://human-settlement.emergency.copernicus.eu/ucdb2024Overview.php)
* [WHO Disease Dataset](https://www.who.int/data)

### Final Dataset
Our final merged dataset combined disaster, climate and socioeconomic data into a unified dataset. This included 16 features relating to environmental and development indicators and was used for our data modelling section. Further details on how this dataset was created can be found in the report.

## Repository Structure
<pre>
Weather-and-Urban-Centre-Emissions-Project
├── Data Exploration/                  # Initial analysis and EDA of MET Office data
├── Data Modelling/                    # PCA, t-SNE and modelling figures 
│   └── Dimensionality Reduction/
│   └── Machine Learning/
├── Data Visualisation/                # Plots and figures from dataset visualisation and exploration
│   └── MET Office Data/
│   └── EMDAT & GDIS/
│   └── GHSL UCDB/
│   └── WHO Disease Data/
├── Datasets/                          # Merged datasets with and without outliers
├── README.md
└── .gitignore
</pre>

## Visualisations
This repository contains additional visualisations that support the analysis presented in the report. These include the following: 
* Exploratory data analysis
* Machine learning and dimensionality reduction
* Model outputs 

## Methods Used
This project applies a range of data science techniques, as depicted in the report:
* Data preparation
  * Dataset merging and aligning multiple sources
  * Interpoltation and aggregation of data
  * Outlier detection and removal 
* Exploratory analysis
  * Pearson correlation analysis
  * Time-series visualisation
* Dimensionality reduction
  * Principal Component Analysis (PCA)
  * t-Distributed Stochastic Neighbour Embedding (t-SNE)
* Machine learning & modelling
  * Clustering (K-Means, Gaussian Mixture Models)
  * Hidden Markov Models (HMMs) for temporal dynamics
  * Regression models (Random Forest, Gradient Boosting, MLP)

## Contributors
* Megan Parfitt
* Ryan Venn
* Bede Paxton
* Finn Slater
* Alex Baousis
* Salah Baaziz
  
