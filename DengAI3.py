from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def addPrevVal(x,data,col_name):
    data[col_name] = data[col_name].shift(x)
    data.fillna(method='bfill', inplace=True)

    return data


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

    sj_need = {'station_avg_temp_c' : 9 , 
                 'station_min_temp_c' : 9, 
                 'station_max_temp_c' : 8, 
                 'reanalysis_min_air_temp_k' : 8}

    iq_need = {'reanalysis_specific_humidity_g_per_kg' : 0, 
                 'station_min_temp_c': 1, 
                 'reanalysis_min_air_temp_k' : 0, 
                 'reanalysis_dew_point_temp_k' : 0}

    for y in range(0,4):
        sj = addPrevVal(sj_need[sj_need.keys()[y]],sj,sj_need.keys()[y])
        iq = addPrevVal(iq_need[iq_need.keys()[y]],iq,iq_need.keys()[y])

    # select features we want
    features_sj = ['station_avg_temp_c', 
                 'station_min_temp_c', 
                 'station_max_temp_c', 
                 'reanalysis_min_air_temp_k']
    features_iq = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_min_air_temp_k', 
                 'reanalysis_dew_point_temp_k', 
                 'station_min_temp_c']
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


print sj_train.describe()
print iq_train.describe()


sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

def get_best_model(train, test, city):
    # Step 1: specify the form of the model

    if city == 'sj':
        model_formula = "total_cases ~ 1 + " \
                        "station_avg_temp_c + " \
                        "station_min_temp_c + " \
                        "station_max_temp_c + " \
                        "reanalysis_min_air_temp_k"
    if city == 'iq':
        model_formula = "total_cases ~ 1 + " \
                        "reanalysis_specific_humidity_g_per_kg + " \
                        "reanalysis_min_air_temp_k + " \
                        "reanalysis_dew_point_temp_k + " \
                        "station_min_temp_c"
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model
    
sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest, 'sj')
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest , 'iq')

figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
sj_train['fitted'] = sj_best_model.fittedvalues
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
iq_train['fitted'] = iq_best_model.fittedvalues
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()

sj_test, iq_test = preprocess_data('./data/dengue_features_test.csv')


# sj_test.fillna(method='bfill', inplace=True)
# iq_test.fillna(method='bfill', inplace=True)

print pd.isnull(sj_test).any()
print pd.isnull(iq_test).any()

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("./data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("./data/benchmark_new3.csv")

plt.show()