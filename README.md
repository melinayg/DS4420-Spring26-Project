# Coral Reef Bleaching Prediction

**DS 4420 - Machine Learning and Data Mining 2 | Spring 2026**

## Overview

This project investigates the environmental drivers of global coral reef bleaching using two machine learning approaches:

1. **Multilayer Perceptron (Python)** — predicts the presence or absence of bleaching from environmental conditions
2. **Bayesian Linear Regression (R)** — identifies which environmental stressors most drive bleaching outcomes with uncertainty estimates

Together the models provide both a predictive risk assessment tool and an inference framework to support coral reef conservation planning.

## Dataset

**Bleaching and Environmental Data for Global Coral Reef Sites (1980–2020)**
van Woesik, R., & Burkepile, D. (2022). BCO-DMO. https://doi.org/10.26008/1912/bco-dmo.773466.2

- 39,422 complete observations after preprocessing
- 62 original columns, 9 features used for modeling
- Binary target: `Bleaching_YN` (1 = any bleaching present, 0 = no bleaching)

## Features

| Feature | Description |
|---|---|
| TSA | Thermal Stress Anomaly at time of survey |
| TSA_DHW | Accumulated thermal stress over 12 weeks |
| SSTA | Sea Surface Temperature Anomaly |
| ClimSST | Climatological baseline sea surface temperature |
| Windspeed | Surface wind speed |
| Turbidity | Water clarity (light penetration) |
| Cyclone_Frequency | Regional cyclone occurrence rate |
| Distance_to_Shore | Distance from reef to nearest coastline |
| Depth_m | Reef survey depth in meters |

## Results

### MLP
- Test Accuracy: **76.8%**
- ROC-AUC: **0.856**
- Bleaching Recall: **0.78** | No Bleaching Recall: **0.77**
- Most important feature: **Turbidity** (permutation importance)

### Bayesian Beta Regression
- Bayesian R²: **~0.04**
- Strongest predictor: **TSA_DHW** (β = 0.16, 95% CI: [0.13, 0.20])
- Depth, Cyclone Frequency, and Windspeed all had **credible intervals above zero**
- Turbidity showed a **slight protective effect** (β = −0.01)
- Full convergence across all chains: **R̂ = 1.00**

## Authors

- Melina Yang - MLP implementation (Python)
- Arshia Mathur - Bayesian Beta Regression (R)
