# MachineLearning

1. Machine learning using scikit-learn.

Scikit-learn models are built and evaluated in python 3.6.9 using multiple regressors. 
Several regressors supporting multi-target regression are selected to explore their performance, they are: RandomForestRegressor, KNeighborsRegressor, MultiTaskElasticNetCV, and MultiTaskLassoCV. 

Due to high computation demand of RandomForestRegressor, we set an iteration number of 20 for RandomForestRegressor and 100 for the rest regressors.

In addition to individual regressors above, two more approaches were employed and their performance were compared with those individual models. First, an ensemble of the four best performing regressors was constructed using StackingRegressor from mlxtend package which support multitarget regression; second, the results from the four best performing regressors were averaged together.

In terms of testing r2 scores, RandomForestRegressor outperforms other models, including the stacked and averaged models.

2. Deep learning.

Deep learning using Keras (2.3.0) functional API with Tensorflow (1.13.1) as backend in python 3.6.8. Talos 0.6.6 was employed for hyperparameter optimization. Environment “tf”.

The parameters to be optimized include number of hidden layers (num_layers), number of neurons for each hidden layer (units), epochs, dropout rates, optimizers, learning rate of the optimizers, and activation algorithms. The loss function was set as mean absolute error. 

The scanning was performed using quantum as random_method, a split_ratio of 0.3 for cross validation, and mean absolute error as the metric. The fraction_limit was set as 0.1 to 0.5 that resulted in 1834 ~ 3000 evaluations. 

The val_loss was used to select the best model.

The optimized deep learning model gave improved r2 score on test data than sklearn regressors including RandomForestRegressor. 

3. "Data_ParFE_32.csv" contains the data for training, evaluation, and testing.
