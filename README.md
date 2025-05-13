# Solar energy forecasting using SEGWO-CNN-LSTM (WIP)

With renewable energies set to play an increasingly important role in the global energy mix, new challenges are emerging. Specifically, their non-controllable nature and dependence on weather conditions are all criteria that require their forecasting. Thus, the quality of each supplier's/balancing manager's forecasts is very important, as they enable, on the one hand, the network balancing manager to maintain network balance without using very costly mechanisms. On the other hand, for the supplier, they reduce the imbalance costs they could create by providing very poor quality forecasts. Thus, it becomes necessary to produce models that can handle highly noisy, non-linear structures such as solar time series. 
The work presented here is a comparative analysis of solar energy production forecasts according to several models:
- An ARIMA baseline (STL decomposition), enabling the results of more complex models to be compared on a reliable basis, and the data to be apprehended in an easier way;
- An LSTM model with hyperparameter optimization by Optuna, with Adam as optimizer (SGD);
- A CNN-LSTM model, to filter LSTM input data (Adam-optimizer). Hyperparameters would be optimized with Optuna;
- A SEGWO-CNN-LSTM model, using a bio-inspired grey wolf pack hunting algorithm, to avoid potential local optimums given by stochastic gradient descent.

*Work in progress*