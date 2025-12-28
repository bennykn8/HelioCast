# HelioCast: Solar Plant Power Forecasting

I aimed to develop a time-series ML model to forecast power production for a solar power plant.

The model performs prediction via multi-horizon forecasting. Specifically, a separate model is trained for each horizon (15min, 30min, 1hr, 2hr, 4hr into the future).

Files of interest in this repository contain:
- **Plant_EDA.ipynb**: exploratory analysis of data trends
- **Plant_Model.ipynb**: data prep/preprocessing, modeling, and model evaluation 

### Business Value/ Use Cases:
Forecasting solar power generation acts as early warning signals for grid operations and enable proactive smart grid management.
A high accuracy ML model adds value to the power market in a few ways:
- Schedule backup generation before solar power drops
- Battery storage optimization (recharge during excess generation and discharge during drops)
- Increase predictability in energy trading market
- Reduce energy waste by rerouting excess generation

Overall, the adoption of a predictive solar power generation models aid in building a smart grid that's backed by stability and reliability.

### Dataset:

https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data 

The dataset used contains information from 2 solar plants in India from mid-May to mid-June of 2020. 
For each plant, data comes in 2 files contain different information: 
- Plant-level sensor data
- Inverter-level power generation data

Data from inverters and sensors are recorded on a 15 minute interval.
Multiple solar panels are connected to an inverter, which converts DC power to AC power.
The plant-level sensor records exogenous information such as ambient temperature, irradiance, and sensor-module temperature.

### Modeling Specifics:

#### **Target Value**:

There are 5 target columns defined in the code by 
`horizons = [1, 2, 4, 8, 16]` corresponding to 15min, 30min, 1hr, 2hr, 4hr forecasting horizons.
For each horizon, a separate regressional model is trained to predict the plant-level power produced within that 15-minute reporting window of the corresponding horizon.

For example, 
- for h = 1 (15mins), we are predicting how much power the plant will yield the next 15 min window from our current time.
- for h = 2 (30mins), we are predicting the power yield of the second window in the future, 15-30 mins from current time.
- for h = 4 (1hr), the power yield of the 4th window in the future: 45min-1hr from current time.

Since the given data only report an accumulating daily yield value, appropriate feature engineering was done to transform the values to a interval-based plant yield.

#### **Feature set:**
- **Environmental & Sensor Features**
  - `AMBIENT_TEMPERATURE`
  - `MODULE_TEMPERATURE`
  - `IRRADIATION`

- **Inverter Availability & Status**
  - `INV_ZERO_FRAC` fraction of inverters producing no AC power
  - `ACTIVE_INV_FRAC` fraction of inverters at timestamp that has recorded data

- **Time-of-Day Encoding**
  - `TOD_SIN` time of day sine 
  - `TOD_COS` time of day cosine

- **Irradiation Temporal Features**
  - `IRRADIATION_LAG1` irradiation lagged 15min
  - `IRRADIATION_LAG2` irradiation lagged 30min
  - `IRRADIATION_ROLLMEAN_1H` rolling mean from last hour
  - `IRRADIATION_ROLLSTD_1H` rolling standard deviation from last hour

- **Module Temperature Temporal Features**
  - `MODTEMP_LAG1`
  - `MODTEMP_ROLLMEAN_1H`

- **AC Power Temporal Features**
  - `AC_PWR_LAG1`
  - `AC_PWR_LAG2`
  - `AC_PWR_LAG4`
  - `AC_PWR_ROLLMEAN_1H`
  - `AC_PWR_ROLLSTD_1H`
  - `AC_PWR_RAMP` 

- **Plant-Level Power Aggregates**
  - `PLANT_DC_PWR`
  - `PLANT_AC_PWR`

- **Inverter-Level AC Power Statistics**
  - `AVG_AC`
  - `STD_AC`
  - `MIN_AC`
  - `MAX_AC`

#### Modeling:
4 candidate baseline models were trained on a training set consisting of the first 80% of the data chronologically.
- Ridge Regression
- Elastic Net
- XGBoost
- LightGBM

Each modeling process involved grid search hyperparameter tuning and 5-fold cross validation.
Models were evaluation based on Mean Absolute Error (MAE), which tells us the average prediction error.