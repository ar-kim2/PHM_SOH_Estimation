# PHM_SOH_Estimation

The project is about how to accurately estimate SOH in the real-world.
To do this, use the following methods: <br>

- The charge/discharge pattern of a battery is represented by four characteristics: <br>
1. entropy of voltage distribution
2. entropy or standard deviation of current distribution
3. total amount of charge
4. average current
- SOH is estimated by using the sequence of these four feature values obtained as the cycle progresses as input to the proposed stacked LSTM Model.

# Datasets
Two datasets were used for the SOH estimation experiment.

- CALCE(Center for Advanced Life Cycle Engineering) dataset
  - Data charged and discharged with constant current every cycle
  - Provided by Maryland University
- Self-collected battery charge/discharge dataset
  - Randomly changes the discharge current in real time within a cycle.
  - Utilizes INR18650 battery model.

# Stacked LSTM Model for estimation SOH

The model for SOH prediction is shown in the following figure.<br><br>
<img src='https://user-images.githubusercontent.com/60689555/233299081-112cde1f-0d2a-4eed-8310-29566170e8ec.png'><br>

- Utilization of LSTM Unit to reflect time series characteristics.
- The model is composed of 2 layers to reflect complex relationships.

# Environment
- Tensorflow

# Reference
- "Online State of Health Estimation of Batteries under Varying Discharging Current Based on a Long Short Term Memory." 2021 15th International Conference on
Ubiquitous Information Management and
Communication (IMCOM). IEEE, 2021.
- "Online Battery SOH Prediction under Intra-Cycle
Variation of Discharge Current and Non-Standard
Charging and Discharging Practices." Annual Conference
of the PHM Society. Vol. 14. No. 1. 2022.
