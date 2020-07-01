# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:28:48 2020
Deep learning using Keras (2.3.0) functional API with Tensorflow (1.13.1) as backend in python 3.6.8. 
Talos 0.6.6 was employed for hyperparameter optimization. Environment “tf”.

The parameters to be optimized include number of hidden layers (num_layers), number of neurons for each hidden layer (units), 
epochs, dropout rates, optimizers, learning rate of the optimizers, and activation algorithms. 
The loss function was set as mean absolute error. 

The scanning was performed using quantum as random_method, a split_ratio of 0.3 for cross validation,
and mean absolute error as the metric. The fraction_limit was set as 0.1~0.5 that resulted in 1834~3000 evaluations. 

Autonomio Talos [Computer software]. (2019). Retrieved from http://github.com/autonomio/talos.
@author: Shuzhang Yang
"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, make_scorer
import joblib
from matplotlib import pyplot as plt
import talos
# from talos.utils import lr_normalizer
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adadelta
# from keras.metrics import MeanAbsoluteError
from scipy import stats


# plot model performance -------------------------------
def plots(y_test, y_pred, par, plot_name):
    '''plot true vs predicted for all parameters (columns in array),
    calculate r2 from pearsonr, output file as plot_name'''
    y_test = y_test
    y_pred = y_pred
    par = par
    np = len(par)
    plot_name = plot_name
    # compute r2 
    r2s = [stats.pearsonr(y_pred[:,i], y_test[:,i])[0]**2 for i in range(y_test.shape[1])]
    s = 2
    a = 0.5
    fig = plt.figure(figsize=(np, 4), dpi=300, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.0, wspace=0.3)
    for i in range(len(par)):
        if (i+1)%2==0:
            j = (i+1)/2 + np/2   ## even index to plot by column wise
        else:
            j = (i+2)/2    ## odd index to plot by column wise
        ax = fig.add_subplot(2, np/2, j)
        ax.scatter(y_test[:, i], y_pred[:, i], edgecolor='k', c="r", s=s, alpha=a, marker=".",
                label=str(par[i]) + " r2=%.3f" % r2s[i])
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.legend(loc="upper left",prop={'size': 6})
        # square subplots
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))
    fig.text(0.5, 0.03, 'Empirical', ha='center', va='center', fontsize=12)
    fig.text(0.015, 0.5, 'Predicted', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.08)
    fig.savefig(plot_name)  


def build_model(x_train, y_train, x_val, y_val, params):
    inputs = Input(shape=(x_train.shape[1],))
    x = inputs
    for i in range(params['num_layers']):
        x = Dense(params['units'], activation=params['activation'])(x)
        x = Dropout(rate=params['dropout'])(x)
    outputs = Dense(y_train.shape[1], activation='linear')(x)
    model = Model(inputs, outputs)
    
    if params['optimizer'] == "Adam":
        opt = Adam(learning_rate=params['lr'])
    if params['optimizer'] == "Nadam":
        opt = Nadam(learning_rate=params['lr'])
    if params['optimizer'] == "Adadelta":
        opt = Adadelta(learning_rate=params['lr'])
        
    model.compile(loss=params['losses'],
                  optimizer=opt,
                  # optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['mae'])
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        workers=-1,          #=========================
                        verbose=0)
    return history, model

# set the parameter space
p = {'num_layers': list(range(1,5)),
     'units':[64, 128, 256, 512],
     'batch_size': [20, 50, 100],
     'epochs': [100, 200, 300, 500],
     'dropout': [0, 0.1, 0.2],
     # 'kernel_initializer': ['uniform','normal'],
     'optimizer': ['Nadam', 'Adam'],
     # 'optimizer': [Nadam, Adam],
     'lr': [0.0001, 0.001, 0.01, 1],
     'losses': ['mean_absolute_error'],
     'activation':['relu', 'elu', 'selu', 'linear']
     # 'last_activation': ['sigmoid', 'linear']
     }

# rng = np.random.RandomState(1)
np.random.seed(543)

# import dataset
files = glob.glob("Data_ParFE*.csv")

# number of targets to predict
nt = 12

experiment_name = 'crispr_target%s' % nt   #=======================

for f in files:
    # f = files[0]
    fname = f.partition(".")[0].partition("FE_")[2]  # substring 32 or 37c
    
    df= pd.read_csv(f)
    par=list(df)[1:(nt+1)] #parameters to predict/plot (perG,perK,phsG,phsK,ampG,ampK,bslG,bslK,trdG,trdK,rG,rK)

    df1 = df.drop(df.columns[[0]], axis=1) # delete the 1st col index
    
    # # check correlogram with seaborn
    # import seaborn as sn
    # sn.pairplot(df1.iloc[0:1000].filter(like='FE'))
    # plt.show()
    
    # split data
    a = df1.to_numpy()  # w/o col names; same as 'a = np.asarray(df1)'
    X = a[:,12:20]   ## Fold Enrichment in 4 gates at 48h & 60h
    y = a[:,0:nt]    ## parameters (perG,perK,phsG,phsK,ampG,ampK,bslG,bslK,trdG,trdK,rG,rK)     

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
#     # sample data --------------------------------------
#    idx = np.random.randint(X_train.shape[0], size=1000)
#    X_train = X_train[idx]
#    y_train = y_train[idx]
    
    
    # run the experiment
    # for random_method in ['uniform_mersenne', 'sobol', 'korobov_matrix', 'quantum']:
    t = talos.Scan(x=X_train,
                   y=y_train,
                   model=build_model,
                   params=p,
                   experiment_name=experiment_name,
                   fraction_limit=0.1,  #fraction from permutation  #=============================
                   random_method='quantum',  #korobov_matrix, quantum, uniform_mersenne, sobol
                   # round_limit=30,
                   val_split=0.3)
    
    # predict with best model -----------------
    best_model = t.best_model(metric='val_loss', asc=True)
    # save best model
    joblib.dump(best_model, 'Mod_KerasTalos_BestModel_%s_%s.pkl' % (experiment_name, fname))      #=======================
    
    y_pred = best_model.predict(X_test, batch_size=100, workers=-1)    #=====================
    
    # # predict from scan object ----------------
    # predict_object = talos.Predict(t)
    # y_pred = predict_object.predict(X_test, 'mae', asc=True)
    
    # plot model performance
    plots(y_test, y_pred, par, 'plot_KerasTalos_Test_%s_%s.pdf' % (experiment_name, fname))  #===============
    plt.close()
    
    # reporting ------------------------------
    analyze_object = talos.Analyze(t)
    # four dimensional bar grid
    analyze_object.plot_bars('batch_size', 'val_mae', 'num_layers', 'optimizer')
    plt.savefig("plot_KerasTalos_batch_%s_%s.pdf" % (experiment_name, fname))
    plt.close()
    analyze_object.plot_bars('epochs', 'val_mae', 'num_layers', 'optimizer')
    plt.savefig("plot_KerasTalos_epochs_%s_%s.pdf" % (experiment_name, fname))
    plt.close()
    analyze_object.plot_bars('dropout', 'val_mae', 'num_layers', 'optimizer')
    plt.savefig("plot_KerasTalos_dropout_%s_%s.pdf" % (experiment_name, fname))
    plt.close()
    analyze_object.plot_bars('activation', 'val_mae', 'num_layers', 'optimizer')
    plt.savefig("plot_KerasTalos_activation_%s_%s.pdf" % (experiment_name, fname))
    plt.close()
    analyze_object.plot_bars('units', 'val_mae', 'num_layers', 'optimizer')
    plt.savefig("plot_KerasTalos_units_%s_%s.pdf" % (experiment_name, fname))
    plt.close()
    analyze_object.plot_bars('lr', 'val_mae', 'num_layers', 'optimizer')
    plt.savefig("plot_KerasTalos_lr_%s_%s.pdf" % (experiment_name, fname))
    plt.close()
    
   
    # # evaluate by cross validation ---------------------------
    # evaluate_object = talos.Evaluate(t)
    # evaluate_object.evaluate(X_train, y_train, folds=10, metric='mae', task='continuous', asc=True)
    
    # deploy & restore --------------------
    talos.Deploy(scan_object=t, model_name='KerasTalos_deploy_%s_%s' % (experiment_name, fname), metric='val_loss', asc=True)
    # crispr = talos.Restore('KerasTalos_deploy_%s.zip' % fname)
    # # make predictions with the restored model
    # y_pred = crispr.model.predict(X_test)
    
