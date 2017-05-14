import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

current_best_sj = 0
current_best_iq = 0
week_no_sj = 0
week_no_iq = 0


def PCA_JOB(x):
    train_features = pd.read_csv('./data/dengue_features_train.csv', index_col=[0,1,2])

    train_labels = pd.read_csv('./data/dengue_labels_train.csv', index_col=[0,1,2])

    # previous Weeks data analysis starts from here
    
    # train_features.insert(2,'prev',0)
    # create previous colomns in the dataframe

    # print train_features

    def addPrevVal(data):
        for i in range(0, len(data.columns.values.tolist())):
            listData = data.columns.values.tolist()[i]
            data[listData + '_prev'] = data[listData].shift(x)

        return data

    sj_train_features = train_features.loc['sj']
    sj_train_labels = train_labels.loc['sj']
    sj_train_features = addPrevVal(sj_train_features)

    # print train_labels

    iq_train_features = train_features.loc['iq']
    iq_train_labels = train_labels.loc['iq']
    iq_train_features = addPrevVal(iq_train_features)

    # print('San Juan')
    # print('features: ', sj_train_features.shape)
    # print('labels  : ', sj_train_labels.shape)

    # print('\nIquitos')
    # print('features: ', iq_train_features.shape)
    # print('labels  : ', iq_train_labels.shape)

    # Remove `week_start_date` string.
    sj_train_features.drop('week_start_date', axis=1, inplace=True)
    iq_train_features.drop('week_start_date', axis=1, inplace=True)

    # checking null values
    # print pd.isnull(sj_train_features).any()

    # (sj_train_features
    #      .ndvi_ne
    #      .plot
    #      .line(lw=0.8))

    # plt.title('Vegetation Index over Time')
    # plt.xlabel('Time')

    sj_train_features.fillna(method='ffill', inplace=True)
    iq_train_features.fillna(method='ffill', inplace=True)

    # check the distribution
    # print('San Juan')
    # print('mean: ', sj_train_labels.mean()[0])
    # print('var :', sj_train_labels.var()[0])

    # print('\nIquitos')
    # print('mean: ', iq_train_labels.mean()[0])
    # print('var :', iq_train_labels.var()[0])

    # sj_train_labels.hist()
    # iq_train_labels.hist()

    sj_train_features['total_cases'] = sj_train_labels.total_cases
    iq_train_features['total_cases'] = iq_train_labels.total_cases

    sj_correlations = sj_train_features.corr()
    iq_correlations = iq_train_features.corr()

    # # plot co relation
    # sj_corr_heat = sns.heatmap(sj_correlations)
    # plt.title('San Juan Variable Correlations')

    # iq_corr_heat = sns.heatmap(iq_correlations)
    # plt.title('Iquitos Variable Correlations')


    print sj_correlations.total_cases.drop('total_cases').sort_values(ascending=False)[:4]
    print iq_correlations.total_cases.drop('total_cases').sort_values(ascending=False)[:4]


    # San Juan
    # (sj_correlations
    #      .total_cases
    #      .drop('total_cases') # don't compare with myself
    #      .sort_values(ascending=False)
    #      .plot
    #      .barh())
    # # plt.show()
    # # Iquitos
    # (iq_correlations
    #      .total_cases
    #      .drop('total_cases') # don't compare with myself
    #      .sort_values(ascending=False)
    #      .plot
    #      .barh())

    # plt.show()
    return sj_correlations.total_cases.drop('total_cases').sort_values(ascending=False)[0],iq_correlations.total_cases.drop('total_cases').sort_values(ascending=False)[0]

for x in range(0, 20):
    best_sj,best_iq = PCA_JOB(x)
    if best_sj > current_best_sj:
        current_best_sj = best_sj
        week_no_sj = x
    if best_iq > current_best_iq:
        current_best_iq = best_iq
        week_no_iq = x

print current_best_sj
print week_no_sj
print current_best_iq
print week_no_iq