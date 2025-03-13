# Weather-Predictor

## Project Abstract

## Introduction
This project aims to construct an agent that can produce accurate weather prediction in New York based on weather patterns of pervious days. This project proposes a probabilistic approach using **Hidden Markov Models (HMMs)** to infer latent weather states, whether there will be rain/snow or not, from historical observational data: daily average dry bulb temperature(°F) and daily average relative humidity. We are also using daily precipitation() and daily snow depth() to classify whether a day counts towards rain/snow or towards no rain/snow. By treating weather evolution as a sequence of hidden variables influenced by observable variables, the model leverages the HMM’s ability to decode **state transitions** and **emission probabilities** from time-series data. The framework is trained on publicly available climate records: [Climate data - New York State. (2022, July 8). Kaggle.] (https://www.kaggle.com/datasets/die9origephit/temperature-data-albany-new-york) , preprocessed into discrete or continuous observation sequences, to learn the dynamics of meteorological states. The resulting agent can probabilistically forecast near-term weather conditions, offering a lightweight, data-driven alternative to complex numerical models. This work highlights the potential of HMMs in modeling environmental systems where unobserved states drive observable outcomes, bridging the gap between interpretable stochastic models and real-world forecasting applications.

---
