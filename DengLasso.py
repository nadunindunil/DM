# -*- coding: utf-8 -*-
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn.linear_model as linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error 

def addPrevVal(x,data,col_name):
    data[col_name] = data[col_name].shift(x)
    data.fillna(method='bfill', inplace=True)

    return data

# 25.0577 MAE for this template
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    sj_need = {'station_avg_temp_c' : 10, 
                 'reanalysis_min_air_temp_k' : 8, 
                 'reanalysis_dew_point_temp_k' : 8, 
                 'reanalysis_specific_humidity_g_per_kg' : 8,
                 'reanalysis_max_air_temp_k': 7,
                 'reanalysis_avg_temp_k': 8,
                 'reanalysis_relative_humidity_percent': 3,
                 'reanalysis_tdtr_k': 10}

    iq_need = {'reanalysis_specific_humidity_g_per_kg' : 0, 
                 'reanalysis_dew_point_temp_k': 0, 
                 'reanalysis_min_air_temp_k' : 0, 
                 'station_min_temp_c' : 1,
                 'reanalysis_air_temp_k': 7,
                 'station_avg_temp_c': 4,
                 'reanalysis_max_air_temp_k': 10,
                 'reanalysis_tdtr_k': 10}

    for y in range(0,8):
        sj = addPrevVal(sj_need[sj_need.keys()[y]],sj,sj_need.keys()[y])
        iq = addPrevVal(iq_need[iq_need.keys()[y]],iq,iq_need.keys()[y])

    # select features we want
    features_sj = ['station_avg_temp_c', 
                 'reanalysis_min_air_temp_k', 
                 'reanalysis_dew_point_temp_k', 
                 'reanalysis_specific_humidity_g_per_kg',
                 'reanalysis_max_air_temp_k',
                 'reanalysis_avg_temp_k',
                 'reanalysis_relative_humidity_percent',
                 'reanalysis_tdtr_k']
    
    features_iq = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'reanalysis_min_air_temp_k', 
                 'station_min_temp_c',
                 'reanalysis_air_temp_k',
                 'station_avg_temp_c',
                 'reanalysis_max_air_temp_k',
                 'reanalysis_tdtr_k']
    
    if labels_path:
        features_sj.append('total_cases')
        features_iq.append('total_cases')

    sj = sj[features_sj]
    iq = iq[features_iq]

    # fill missing values
    sj.fillna(method='bfill', inplace=True)
    iq.fillna(method='bfill', inplace=True)
    
    return sj, iq

#############################################################################################

sj_train, iq_train = preprocess_data('./data/dengue_features_train.csv',
                                    labels_path="./data/dengue_labels_train.csv")


#print sj_train.describe()
#print iq_train.describe()


sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

def get_best_model(train, city):
 
    X = train.drop('total_cases', axis=1)
    y = train["total_cases"]
#    lasso = Lasso(random_state=0)
#    alphas = 10 ** np.arange(-8, 3, dtype=np.float64)
##    scores = list()
##    scores_std = list()
#    best_alpha = []
#    best_score = -1000
#    
#    n_folds = 3
#    
#    for alpha in alphas:
#        lasso.alpha = alpha
#        lasso.fit(X,y)
##        score = lasso.score(X,y)
#        this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
##        scores.append(np.mean(this_scores))
##        scores_std.append(np.std(this_scores))
#        
#        score = np.mean(this_scores)
#        print score
#        if score > best_score:
#            best_alpha = alpha
#            best_score = score
#            
#    print('best alpha = ', best_alpha)
#    print('best score = ', best_score)
    
#    scores, scores_std = np.array(scores), np.array(scores_std)
#    
#    plt.figure().set_size_inches(8, 6)
#    plt.semilogx(alphas, scores)
#    
#    # plot error lines showing +/- std. errors of the scores
#    std_error = scores_std / np.sqrt(n_folds)
#    
#    plt.semilogx(alphas, scores + std_error, 'b--')
#    plt.semilogx(alphas, scores - std_error, 'b--')
#    
#    # alpha=0.2 controls the translucency of the fill color
#    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
#    
#    plt.ylabel('CV score +/- std error')
#    plt.xlabel('alpha')
#    plt.axhline(np.max(scores), linestyle='--', color='.5')
#    plt.xlim([alphas[0], alphas[-1]])
    
    #ridgereg = Ridge(alpha=1e-100, max_iter= 100000, solver="lsqr")
    #fitted_model = ridgereg.fit(X,y)
     
    #model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    model = Lasso(selection="random", normalize=True, random_state=7, alpha = 1e-30 )
    fitted_model = model.fit(X,y)
    
    return fitted_model
    
    
    
sj_best_model = get_best_model(sj_train, 'sj')
iq_best_model = get_best_model(iq_train, 'iq')

figs, axes = plt.subplots(nrows=2, ncols=1)

sj_X = sj_train.drop('total_cases', axis=1)
iq_X = iq_train.drop('total_cases', axis=1)

sj_pred = sj_best_model.predict(sj_X).astype(int)
iq_pred = iq_best_model.predict(iq_X).astype(int)

# plot sj
sj_train['fitted'] = sj_pred
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
iq_train['fitted'] = iq_pred
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

print "SJ mae : ",
print mean_absolute_error(sj_train['total_cases'], sj_train['fitted'])
print "IQ mae : ",
print mean_absolute_error(iq_train['total_cases'], iq_train['fitted'])

sj_test, iq_test = preprocess_data('./data/dengue_features_test.csv')


# sj_test.fillna(method='bfill', inplace=True)
# iq_test.fillna(method='bfill', inplace=True)

#print pd.isnull(sj_test).any()
#print pd.isnull(iq_test).any()

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("./data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("./data/lasso_2iq.csv")

plt.show()

