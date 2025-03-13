# Weather-Predictor

## Project Abstract

## Introduction
This project aims to construct an agent that can produce accurate rain or snow prediction in New York based on weather patterns of pervious days. This project proposes a probabilistic approach using **Hidden Markov Models (HMMs)** to infer latent weather states, whether there will be rain/snow or not, from historical observational data: daily average dry bulb temperature(°F) and daily average relative humidity(%). We are also using daily precipitation(inch) and daily snow depth(inch) to classify whether a day counts towards rain/snow or towards no rain/snow. By treating weather evolution as a sequence of hidden variables (rain/snow or no rain/snow) influenced by observable variables (temperature and humidity), the model leverages the HMM’s ability to decode **state transitions** and **emission probabilities** from time-series data. The framework is trained on publicly available climate records: [Climate data - New York State. (2022, July 8). Kaggle.] (https://www.kaggle.com/datasets/die9origephit/temperature-data-albany-new-york), preprocessed into discrete observation sequences. The resulting agent can probabilistically forecast near-term weather conditions, offering a lightweight, data-driven alternative to complex numerical models. This work highlights the potential of HMMs in modeling environmental systems where unobserved states drive observable outcomes, bridging the gap between interpretable stochastic models and real-world forecasting applications.

---

* **P**erformance measure: accuracy of its predictions of whether it is going to rain or snow or not.
* **E**nvironment: Only able to see the input statement and the training data.
* **A**ctuators: the input cell (sequence of whether) and output cell (rain/snow or not tommorrow).
* **S**ensors: the input cell.
This is a goal based model, with the only goal to find the probability of rain/snow for the next day given the weather sequence of the previous day(s), and return a output based on that value. This is a model based agent. It trains on the dataset once and will answer only from its training.

##  Data Exploration and Preprocessing Step

### We are using the dataset daily_data.csv

---

There are 2668 number of observations in this training set, each with 19 features. The datas are from January 1st, 2015 to May 31, 2022, which some days missing but we can ignore that since it is a large dataset. The features are: *STATION,DATE,REPORT_TYPE,SOURCE,BackupElements,BackupElevation,BackupEquipment,BackupLatitude,BackupLongitude,BackupName,DailyAverageDewPointTemperature,DailyAverageDryBulbTemperature,DailyAverageRelativeHumidity,DailyAverageSeaLevelPressure,DailyAverageStationPressure,DailyAverageWetBulbTemperature,DailyAverageWindSpeed,DailyCoolingDegreeDays,DailyDepartureFromNormalAverageTemperature,DailyHeatingDegreeDays,DailyMaximumDryBulbTemperature,DailyMinimumDryBulbTemperature,DailyPeakWindDirection,DailyPeakWindSpeed,DailyPrecipitation,DailySnowDepth,DailySnowfall,DailySustainedWindDirection,DailySustainedWindSpeed,Sunrise,Sunset,WindEquipmentChangeDate*, but we will not be need ing all of them. 
We only need the 12th, 13th, 25th, and 26th feature of each observation. The fist 2568 observations are going to be the training data of this project and the last 100 be the validation set. 
The data set is realiable since it was taken directly from a national weather station in Albany, New York.

---
