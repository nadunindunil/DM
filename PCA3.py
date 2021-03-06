import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import csv

# PCA3 describe about the correlation of each feature 
# with totall-cases in time of n given number of weeks

features = pd.read_csv('./data/dengue_features_train.csv', index_col=[0,1,2])
features.drop('week_start_date', axis=1, inplace=True)
file = open('./output/pca3_new.csv','w')
fieldnames = ['city', 'feature','best score', 'week']
writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator = '\n')
writer.writeheader()

def PCA_JOB(x,col_name):
    train_features = pd.read_csv('./data/dengue_features_train.csv', index_col=[0,1,2])

    train_labels = pd.read_csv('./data/dengue_labels_train.csv', index_col=[0,1,2])

    # previous Weeks data analysis starts from here
    
    # train_features.insert(2,'prev',0)
    # create previous colomns in the dataframe

    # print train_features

    def addPrevVal(data):
        # for i in range(0, len(data.columns.values.tolist())):
        #     listData = data.columns.values.tolist()[i]
        #     data[listData + '_prev'] = data[listData].shift(x)
        data[col_name] = data[col_name].shift(x)
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

    sj_train_features.fillna(method='ffill', inplace=True)
    iq_train_features.fillna(method='ffill', inplace=True)

    # sj_train_labels.hist()
    # iq_train_labels.hist()

    sj_train_features['total_cases'] = sj_train_labels.total_cases
    iq_train_features['total_cases'] = iq_train_labels.total_cases

    sj_correlations = sj_train_features.corr()
    iq_correlations = iq_train_features.corr()

    # plt.show()
    return sj_correlations.total_cases.drop('total_cases')[col_name], iq_correlations.total_cases.drop('total_cases')[col_name]

for y in (features.columns.values.tolist()):
    current_best_sj = 0
    current_best_iq = 0
    week_no_sj = 0
    week_no_iq = 0

    for x in range(0, 11):
        best_sj,best_iq = PCA_JOB(x , y)

        if best_sj > current_best_sj:
            current_best_sj = best_sj
            week_no_sj = x
        if best_iq > current_best_iq:
            current_best_iq = best_iq
            week_no_iq = x

    print "\ncurrent best for sj " + str(y) + " : " + str(current_best_sj)
    print "in week : " + str(week_no_sj) 
    print "current best for iq " + str(y) + " : " + str(current_best_iq)
    print "in week : " + str(week_no_iq)
    
    writer.writerow({"city" : "sj" , "feature" : str(y), "best score" : current_best_sj, "week" : week_no_sj})
    writer.writerow({"city" : "iq" , "feature" : str(y), "best score" : current_best_iq, "week" : week_no_iq})

file.close()