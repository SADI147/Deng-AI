# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:39:21 2020

@author: Sourav
"""

import pandas as pd
import numpy as np
import seaborn as sns

data_features = pd.read_csv('dengue_features_train.csv')
data_labels = pd.read_csv('dengue_labels_train.csv')

data_labels = data_labels.drop(['city', 'year', 'weekofyear'], axis = 1)

final = pd.concat([data_features, data_labels], axis = 1)
final = final.drop(['city', 'year', 'weekofyear', 'week_start_date'], axis = 1)

#DROP nan
final.isnull().sum()
final = final.fillna(final.ffill())
final.isnull().sum()

#
##get correlations of each features in dataset
#corrmat = final.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
##plot heat map
#g=sns.heatmap(final[top_corr_features].corr(),annot=True,cmap="RdYlGn")



ctk = ["precipitation_amt_mm", "station_avg_temp_c", 'ndvi_ne', 'ndvi_nw', 
       'ndvi_se', 
       'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
       'reanalysis_dew_point_temp_k', 
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 
       'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 
       'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 
       'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'station_max_temp_c', 
       'station_min_temp_c', 'station_precip_mm']


#------------------------------------------------------------------------------
#APPLY LOG TRANSFORMATION
#------------------------------------------------------------------------------
#df1 = final['total_cases']
#df2 = []
#
#
#for i in df1:
#    if i>0:
#        df2.append(np.log(i))
#    else:
#        df2.append(0)
#
#df2 = pd.DataFrame(df2)


X = final[ctk]  #independent columns
Y = final['total_cases']    #target column i.e total_cases


#------------------------------------------------------------------------------
#STANDARD SCALING
#------------------------------------------------------------------------------
#
#
#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#scaled_x = ss.fit_transform(X)
#Y = (Y-Y.mean())/Y.std()
#scaled_x =pd.DataFrame(scaled_x)


#------------------------------------------------------------------------------
#TRAIN TEST SPLIT 
#------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split as tts
train_x, test_x, train_y, test_y = tts(X, Y, test_size = 0.2,
                                       random_state = 15)

##
#tr_size = 0.8 * len(X)
#tr_size = int(tr_size)
#
#X_train = X.values[0 : tr_size]
#X_test = X.values[tr_size : len(X)]
#
#Y_train = Y.values[0 : tr_size]
#Y_test = Y.values[tr_size : len(Y)]
#



#------------------------------------------------------------------------------
#FIT THE REGRESSION MODEL
#------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn import metrics

std_reg = LinearRegression()
std_reg.fit(train_x, train_y)

r2_train = std_reg.score(train_x, train_y)
r2_test  = std_reg.score(test_x, test_y)

print(r2_train)
print(r2_test)

y_pred = std_reg.predict(test_x)

np.round(metrics.mean_absolute_error(y_pred, test_y), 5)
np.sqrt(metrics.mean_squared_error(y_pred, test_y))

print(std_reg.coef_)

#------------------------------------------------------------------------------
#ELIMINATE FEATURES WITH NEGATIVE AND LEAST POSITIVE COEFFICIENT VALUE
#------------------------------------------------------------------------------
new = ['ndvi_nw', 'reanalysis_air_temp_k', 'reanalysis_max_air_temp_k',
       'reanalysis_specific_humidity_g_per_kg']
X = X[new] 

train_x, test_x, train_y, test_y = tts(X, Y, test_size = 0.2,
                                       random_state = 15)

std_reg.fit(train_x, train_y)

r2_train = std_reg.score(train_x, train_y)
r2_test  = std_reg.score(test_x, test_y)

print(r2_train)
print(r2_test)

y_pred = std_reg.predict(test_x)

np.round(metrics.mean_absolute_error(y_pred, test_y), 5)
np.sqrt(metrics.mean_squared_error(y_pred, test_y))



#------------------------------------------------------------------------------
#DENGUE TEST FEATURES PREDICTIONS
#------------------------------------------------------------------------------

test_data = pd.read_csv('dengue_features_test.csv')


test_data_predict = test_data[new]

test_data_predict = test_data_predict.fillna(test_data_predict.ffill())
#test_data_predict = ss.fit_transform(test_data_predict)
test_data_predict.isnull().sum()

predict_y = std_reg.predict(test_data_predict)

y_predict=[]
for i in predict_y:
    y_predict.append(i)
        
for k in y_predict:
    if k < 0:
        y_predict = pd.DataFrame(y_predict, columns = ['total_cases']).replace(to_replace = k, method = 'ffill')

    
y_predict = np.ceil(y_predict)

output = pd.concat([test_data['city'], test_data['year'], 
                    test_data['weekofyear'],pd.DataFrame(y_predict, columns = ['total_cases'])],
                    axis = 1)

output.dtypes

output = output.astype({'total_cases': 'int64'})
output.to_csv('Dengue_Predictions.csv', index = False)    

