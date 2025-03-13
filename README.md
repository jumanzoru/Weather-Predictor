# Weather-Predictor

## Project Abstract

## Introduction
This project aims to construct an agent that can produce accurate predictions of whether it will rain tomorrow in New York based on today's temperature and humidity. This project proposes a probabilistic approach using **Hidden Markov Models (HMMs)** to infer latent weather states, whether there will be rain/snow or not, from historical observational data: daily average dry bulb temperature(°F) and daily average relative humidity(%). We are also using daily precipitation(inch) and daily snow depth(inch) to classify whether a day counts towards rain/snow or towards no rain/snow. By treating weather evolution as a sequence of hidden variables (rain/snow or no rain/snow) influenced by observable variables (temperature and humidity), the model leverages the HMM’s ability to decode **state transitions** and **emission probabilities** from time-series data. The framework is trained on publicly available climate records: [Climate data - New York State. (2022, July 8). Kaggle.] (https://www.kaggle.com/datasets/die9origephit/temperature-data-albany-new-york), preprocessed into discrete observation sequences. The resulting agent can probabilistically forecast near-term weather conditions, offering a lightweight, data-driven alternative to complex numerical models. This work highlights the potential of HMMs in modeling environmental systems where unobserved states drive observable outcomes, bridging the gap between interpretable stochastic models and real-world forecasting applications.

---

* **P**erformance measure: accuracy of its predictions of whether it is going to rain or snow or not.
* **E**nvironment: Only able to see the input statement and the training data.
* **A**ctuators: the input cell (sequence of whether) and output cell (rain/snow or not tommorrow).
* **S**ensors: the input cell.

* This is a goal based model, with the only goal to find the probability of rain/snow for the next day given the weather sequence of the previous day(s), and return a output based on that value. This is a model based agent. It trains on the dataset once and will answer only from its training.

##  Data Exploration and Preprocessing Step

### We are using the dataset daily_data.csv

---

There are 2668 number of observations in this training set, each with 19 features. The datas are from January 1st, 2015 to May 31, 2022, which some days missing but we can ignore that since it is a large dataset. The features are: 

> STATION,DATE,REPORT_TYPE,SOURCE,BackupElements,BackupElevation,BackupEquipment,BackupLatitude,BackupLongitude,BackupName,DailyAverageDewPointTemperature,DailyAverageDryBulbTemperature,DailyAverageRelativeHumidity,DailyAverageSeaLevelPressure,DailyAverageStationPressure,DailyAverageWetBulbTemperature,DailyAverageWindSpeed,DailyCoolingDegreeDays,DailyDepartureFromNormalAverageTemperature,DailyHeatingDegreeDays,DailyMaximumDryBulbTemperature,DailyMinimumDryBulbTemperature,DailyPeakWindDirection,DailyPeakWindSpeed,DailyPrecipitation,DailySnowDepth,DailySnowfall,DailySustainedWindDirection,DailySustainedWindSpeed,Sunrise,Sunset,WindEquipmentChangeDate


But we will not be needing all of them. 
We only need the 12th, 13th, 25th, and 26th feature of each observation. The fist 2568 observations are going to be the training data of this project and the last 100 be the validation set. 
The data set is realiable since it was taken directly from a national weather station in Albany, New York.

---

There are 4 columns that we are using are: 
* Daily average dry bulb temperature (°F) (column 12)
* Daily average relative humidity (%) (column 13)
* Daily Precipitation (inch) (column 25)
* Daily Snow Depth (inch) (column 26)

---

### Binarize features:
* Daily average dry bulb temperature: calculated median for all data points in the training set: 51.0. Any data point with a temperature higher or equal to this threshold will be classified as *high*, others are *low*
* Daily average relative humidity: calculated median for all data points in the training set: 66.0. Any data point with a humidity higher or equal to this threshold will be classified as *high*, others are *low*
* Daily Precipitation and Daily Snow Depth: any data point that has a precipitation greater than 0 OR has a precipitation greater than 0 will be classified as *rain/snow*, others are *no rain/snow*

* The median calculation step is done in **getting thresholds for binarylization.ipynb**
* ![image](https://github.com/user-attachments/assets/7f7f4236-7554-4d0c-a6d1-fd7204b4f998)


---

## Objective

### Given:
Today's temperature and humidity

### Return:
Is it going to rain/snow tomorrow

---

## Assumption: 
Weather transitions are Markovian (tomorrow’s rain/snow or not depends only on today’s).

---

## Method

**Use Forward Algorithm for calculating today’s state, then use the predicted state of today to predict tomorrow’s state**

### Note: 
We are not taking Rain/Snow or not from the user input, we are predicting it. Rain/Snow or not is an hidden variable! This is because sometimes this piece of information could be missing and we also believe that the **Forward-Backward Algorithm** on Hidden Markov Models will produce a more accurate result than just directly returning the likelihood of Rain/Snow or not from data.

### But in order to use the Forwad Algorithm, we need to set up a HMM

![Image](diagram.jpg)

* Hidden States: The weather (rain/snow, no rain/snow).

* Observations: The measured data (temperature, humidity).

#### Model Parameters:

An HMM is defined by three matrices:

1. Transition Matrix A: Probability of moving from state i to j.
  * A_ij = P(next state = j∣current state = i)​, (shape: 2 x 2)
  * There are 2 possible states: [Rain/Snow, no Rain/Snow]

2. Emission Matrix B: Probability of observing k given state i.
  * B_ik = P(observation = k∣state = i), (shape: 2 x 4)
  * There are 4 possible observations: [High temp High humidity, High temp Low humidity, Low temp High Humidity, Low temp, Low Humidity]
3. Initial State Distribution: Probability of starting in state i.
  * init_i = P(initial state = i), (shape: 2)


#### Computations of the three matrices
Compute the transition matrix A, emission matrix B, and initial distribution π directly from counts.

* A_ij = Number of transitions from state i to j / Total transitions from State i

  * Example: If no Rain/Snow occurs 100 times and transitions to Rain/Snow 20 times:
    > A_Rain/Snow, no Rain/Snow = 20 /100 = 0.2

* B_ik = Count of observation k in state i / Total observations in state i

* π_i = Frequency of state i = Count of state i / Total number of data

### Forward Algorithm for Today’s State

1. Compute the probability of being in state i today given today’s observation:
   * This is the filtered state distribution given today’s observation. Using Bayes’ theorem:
     > P(in state i today | today’s observation) = P(today’s observation | in state i today) * P(in state i today) / P(today’s observation)
   * Let α_observation,i = B_i,observation * π_i, and P(today’s observation) = sum over j of B_j,observation * π_j
     > P(in state i today | today’s observation) = α_observation,i / sum over j of α_observation,j

2. Predict Tomorrow’s State
   * Use the transition matrix A to compute the probability of transitioning to state j tomorrow:
     > P(state j tomorrow | today’s observation) = sum over i of P(in state i today | today’s observation) * P(state j tomorrow | state i today)
   * P(in state i today | today’s observation) are calculated in part 1 and P(state j tomorrow | state i today) is just A_ij
     > P(state j tomorrow | today’s observation) = sum over i of (α_observation,i / sum over j of α_observation,j) * P(state j tomorrow | state i today)

3. Extract the probability of "rain/snow tomorrow":
   * P(rain/snow tomorrow | today’s observation)
   * if < 0.5, then return "most likely no rain or snow tomorrow"
   * if >= 0.5, then return "most likely will rain or snow tomorrow"

---

## Future Feature Expansions
For future improvements:
* the agent can allow users to input a whole sequence of observations (e.g., past 3 days) for more robust state inference. A longer sequence would improve the accuracy of a Forward Algorithm on HMMs.
* Adding more data to the training data would also increase the acurracy.
* In addition, Using more recent data would be better. This is because the climate is changing hence behaviors tens of years before are less likely suitable for the behaviors of weather today.
* We should also check the observation that the user inputted. If the observation is very rare, for example having a very extreme temperature that the training data lacks, then refuse to make the prediction and explain to the user.
* Another way improvement would be not to binarize the data (and user input) and use Gaussian HMM

---

## Conclusion
tbd

#### Potential Issues:
tbd

#### Drawbacks:
tbd

# Instructions
**To run the agent, download the WeatherPredictor.ipynb file and the daily_data.csv file in the SAME directory. If you are using google collab, you will need to drag the csv file to the "file" location in google collab and overwrite the file location when reading it.**

---

## Contributors
* Tom Tang
* Guan Huang-Chen
* Xueheng Zhou
* Jefferson Umanzor-Urrutia
* [Iron486 (dataset owner)](https://www.kaggle.com/die9origephit)
