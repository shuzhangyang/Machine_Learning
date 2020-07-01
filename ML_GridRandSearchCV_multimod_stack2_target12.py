# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:09:25 2020
# adapted from: http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/

@author: shuzh
"""

import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

class EstimatorSelectionHelper:
    
    def __init__(self, models, params, searcher=GridSearchCV):
        self.models = models
        self.params = params
        self.searcher = searcher
        self.keys = models.keys()
        self.searches = {}
        if searcher == GridSearchCV:
            self.searcher_name = 'GridSearchCV'
        elif searcher == RandomizedSearchCV:
            self.searcher_name = 'RandomizedSearchCV'
    
    def fit(self, X, y, **search_kwargs):
        for key in self.keys:
            print('Running %s for %s.' % (self.searcher_name, key))
            model = self.models[key]
            params = self.params[key]
            if self.searcher_name == 'RandomizedSearchCV':
                if key == 'RandomForestRegressor':
                    search = self.searcher(model, params, n_iter=20, **search_kwargs)
                else:
                    search = self.searcher(model, params, n_iter=100, **search_kwargs)
            else:
                search = self.searcher(model, params, **search_kwargs)
            search.fit(X, y)
            self.searches[key] = search
        print('Done.')
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, search in self.searches.items():
            frame = pd.DataFrame(search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df
    
    def bests(self):
        best_scores = {}
        best_params = {}
        best_estimators = {}
        for name, search in self.searches.items():
            best_scores[name] = search.best_score_
            best_params[name] = search.best_params_
            best_estimators[name] = search.best_estimator_

        best_scores_df = pd.DataFrame([best_scores]).T
        best_scores_df.columns = ['best_score']
        best_params_df = pd.DataFrame([best_params]).T
        best_params_df.columns = ['best_params']
        best_df = pd.merge(best_scores_df, best_params_df, left_index=True, right_index=True)
        return best_df, best_estimators

from mlxtend.regressor import StackingRegressor
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskLassoCV, Ridge
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, make_scorer
import joblib
from matplotlib import pyplot as plt


models = { 
    # 'HistGradientBoostingRegressor': HistGradientBoostingRegressor(), #not multioutput
    'RandomForestRegressor': RandomForestRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'MultiTaskElasticNetCV': MultiTaskElasticNetCV(),
    'MultiTaskLassoCV': MultiTaskLassoCV()
    # 'Ridge': Ridge()
}

paramsR = { 
    # 'HistGradientBoostingRegressor': {'max_iter': stats.loguniform(1e2, 1e4),
    #                                   'learning_rate': stats.loguniform(1e-2,1)},
    'RandomForestRegressor': {'n_estimators': stats.randint(1e2, 1e3),   #===========================
                                'criterion': ['mae']},
    'KNeighborsRegressor':  {'n_neighbors': stats.randint(2,100),
                              'weights': ['uniform', 'distance']},
    'MultiTaskElasticNetCV': {'l1_ratio': stats.loguniform(1e-3, 1), 
                               'eps': stats.loguniform(1e-5, 1e-1),
                               'n_alphas': stats.randint(10,10000)},
    'MultiTaskLassoCV': {'eps': stats.loguniform(1e-5, 1e-1),
                          'n_alphas': stats.randint(10,10000)}
    # 'Ridge':{'alpha': stats.loguniform(1e-2, 1e2)}
    }

# paramsG = { 
    # 'HistGradientBoostingRegressor': {'max_iter': stats.loguniform(1e2, 1e4),
    #                                   'learning_rate': stats.loguniform(1e-2,1)},
    # 'RandomForestRegressor': {'n_estimators': [100, 200, 300, 400, 500],
                              # 'criterion': ['mse', 'mae']},
    # 'KNeighborsRegressor':  {'n_neighbors': [5, 10, 20, 50, 100],
                         # 'weights': ['uniform', 'distance']},
    # 'MultiTaskElasticNetCV': {'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9], 
    #                            'eps': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #                            'n_alphas': [50, 100, 200, 500, 1000]},
    # 'MultiTaskLassoCV': {'eps': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    #                       'n_alphas': [50, 100, 200, 500, 1000]}
    # 'RidgeCV':{'alphas': [0.1, 1, 10]}
    # }

def multiplot(name, estimator, X_test, y_test, targets):
    '''plot the quality for trained estimator with name on test data
    for all targets (perG, perK, phsG, phsK, ampG, ampK, bslG, bslK ...'''
    name = name    # estimator name
    est = estimator
    X_test = X_test
    y_test = y_test
    par = targets
    np = len(par)
    # compute r2 scores for multioutpus
    y_pred = est.predict(X_test)
    r2s = r2_score(y_test, y_pred, multioutput='raw_values')
    
    # plot
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
    fig.savefig('plot_'+name+'.pdf')  #======================
 
# os.chdir("C:/Users/shuzh/Documents/Scripts/Crispr/ML/")

# rng = np.random.RandomState(1)

# import dataset
files = glob.glob("Data_ParFE*.csv")
#f = [i for i in files if "32" in i][0]  # select file by pattern

# number of targets to predict (perG,perK,phsG,phsK,ampG,ampK,bslG,bslK,trdG,trdK,rG,rK) 
nt = 12    #===================================

for f in files:
    fname = f.partition(".")[0].partition("FE_")[2]  # substring 32 or 37c
    
    df= pd.read_csv(f)
    par=list(df)[1:(nt+1)] # parameters to plot

    df1 = df.drop(df.columns[[0]], axis=1) # delete the 1st col index
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
    
    #multioutput weight (trend & damping 0.1)
    if nt>8:
        multioutput_wt = [1]*8 + [0.1]*(nt-8)
    else:
        multioutput_wt = [1]*nt
    
    scorer = make_scorer(r2_score, multioutput=multioutput_wt)
    helper = EstimatorSelectionHelper(models, paramsR, searcher=RandomizedSearchCV)  #==================
    helper.fit(X_train, y_train, scoring=scorer, n_jobs=-1)
    
    res, estimators = helper.bests()
    res.to_csv('BestScoreParams_%s_%s_target%s.csv' % (helper.searcher_name, fname, nt)) #=======================
    
    summary = helper.score_summary()
    summary.to_csv('Score_Summary_%s_%s_target%s.csv' % (helper.searcher_name, fname, nt)) #=======================
    
    # save best models; score; plot
    for name, est in estimators.items():
        joblib.dump(est, 'Mod_'+name+'_%s_target%s.pkl' % (fname, nt))      #=======================
        multiplot(name+'_%s_target%s' % (fname, nt), est, X_test, y_test, par)
        plt.close()
    
    # stack the best models mlxend.regressor.StackingRegressor ---------------------
    stack_regressor = StackingRegressor(
        regressors=[
                    RandomForestRegressor(n_estimators=res.at['RandomForestRegressor', 'best_params']['n_estimators'],
                                           criterion=res.at['RandomForestRegressor', 'best_params']['criterion']),
                    KNeighborsRegressor(n_neighbors=res.at['KNeighborsRegressor','best_params']['n_neighbors'],
                                        weights=res.at['KNeighborsRegressor','best_params']['weights']),
                    MultiTaskElasticNetCV(res.at['MultiTaskElasticNetCV','best_params']['eps'],
                                          res.at['MultiTaskElasticNetCV','best_params']['l1_ratio'],
                                          res.at['MultiTaskElasticNetCV','best_params']['n_alphas']),
                    MultiTaskLassoCV(res.at['MultiTaskLassoCV','best_params']['eps'], 
                                     res.at['MultiTaskLassoCV','best_params']['n_alphas'])],
        meta_regressor=Ridge())
    stack_regressor.fit(X_train, y_train)
    #save model
    joblib.dump(stack_regressor, 'Stack_regressor_%s_target%s.pkl' % (fname, nt))   #===================
    #plot results
    multiplot('Stack_regressor_%s_target%s' % (fname, nt), stack_regressor, X_test, y_test, par)
    plt.close()
    
    # manual averaging across multiple model predictions ---------------------------
    y_preds = []
    for name, est in estimators.items():
        y_preds.append(est.predict(X_test))
    y_pred_mean = np.mean(y_preds, axis=0)
    
    r2sm = [stats.pearsonr(y_test[:,i], y_pred_mean[:,i])[0]**2 for i in range(y_test.shape[1])]
    
    # plot manual mean
    s = 2
    a = 0.5
    fig = plt.figure(figsize=(nt, 4), dpi=300, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.0, wspace=0.3)
    for i in range(len(par)):
        if (i+1)%2==0:
            j = (i+1)/2 + nt/2   ## even index to plot by column wise
        else:
            j = (i+2)/2    ## odd index to plot by column wise
        ax = fig.add_subplot(2, nt/2, j)
        ax.scatter(y_test[:, i], y_pred_mean[:, i], edgecolor='k', c="r", s=s, alpha=a, marker=".",
                label=str(par[i]) + " r2=%.3f" % r2sm[i])
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.legend(loc="upper left",prop={'size': 6})
        # square subplots
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))
        
    fig.text(0.5, 0.03, 'Empirical', ha='center', va='center', fontsize=12)
    fig.text(0.015, 0.5, 'Predicted', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.08)
    fig.savefig('plot_Mean_%s_target%s.pdf' % (fname, nt))  #======================
    plt.close()
    