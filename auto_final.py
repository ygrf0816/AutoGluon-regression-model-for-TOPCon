from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autogluon.features.generators import AutoMLPipelineFeatureGenerator as generator
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import time
#from func import draw

data_length=13
train_test = pd.read_csv('dataset/regression.csv')
train_test = train_test.sample(frac=1).reset_index(drop=True)
column = train_test.columns

x = train_test.iloc[:,0:data_length]
scaler = StandardScaler().fit(x) # Whether to normalize the data
x=pd.DataFrame(scaler.transform(x))
y = train_test.iloc[:,data_length:data_length+6] 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,shuffle=True)

train = np.concatenate((x_train,y_train),axis=1)
test = np.concatenate((x_test,y_test),axis=1)
train = pd.DataFrame(train)
train.columns = column
test = pd.DataFrame(test)
test.columns = column

train_data = TabularDataset(train)
test_data = TabularDataset(test)

label = ['Vm','Im','Voc','Jsc','FF','Eff']
lt = ['Vm','Im','Voc','Jsc','FF','Eff']

# Whether to train the model, set to True to train the model
is_train = True
# is_train = False

result = pd.DataFrame()

for item in lt:
    need_to_drop = []  # Initialize need_to_drop as an empty list

    current_label = item  # Assign item to current_label to avoid overwriting the original label list
    need_to_drop = [x for x in label if x != item]
    temp = train.drop(columns=need_to_drop)
    temp = TabularDataset(temp)
    if is_train==True:
        f = open(('output/log_'+current_label+str(time.time())+'.txt'),'w')
        print('The '+item+' model is training.')
        predictor = TabularPredictor(
        label=current_label,path=('Models//final_v3//'+current_label),verbosity=2,
        problem_type='regression',eval_metric='rmse').fit(
            train_data=temp,presets='best_quality')
        TabularPredictor.leaderboard(predictor,silent=True).to_csv('output/leaderboard_'+current_label+'.csv',index=False)
        # print(TabularPredictor.fit_summary(predictor,verbosity=2,show_plot=True),file=f)
        TabularPredictor.feature_importance(predictor,data=temp).to_csv('output/importance_'+current_label+'.csv')

        predict = predictor.predict(test_data)
        r2_test=r2_score(test[item],predict) 
        mae_test=mean_absolute_error(test[item],predict)
        mse_test=mean_squared_error(test[item],predict)

        predict = predictor.predict(train_data)
        r2_train=r2_score(train[item],predict) 
        mae_train=mean_absolute_error(train[item],predict)
        mse_train=mean_squared_error(train[item],predict)
        result = pd.concat([result,predict],axis=1)
        a=np.array(test[item])
        b=np.array(predict)
        # Old comparison plotting method
        """ draw(a[0:99],b[0:99],title=(item+' '+'Prediction'),
                path='figure/auto/pred_'+item+'.png',mode='predict') """ 
        print('r2 test='+str(r2_test)+',mae test='+str(mae_test)+',mse test='+str(mse_test),file=f)
        print('r2 train='+str(r2_train)+',mae train='+str(mae_train)+',mse test='+str(mse_train),file=f)
        f.close()
    else:
        print('Load '+item+' model')
        f = open(('output/log_'+current_label+str(time.time())+'.txt'),'w')
        predictor = TabularPredictor.load('Models//final_v3//'+current_label)

        predict = predictor.predict(test_data)
        r2_test=r2_score(test[item],predict) 
        mae_test=mean_absolute_error(test[item],predict)
        mse_test=mean_squared_error(test[item],predict)

        predict = predictor.predict(train_data)
        r2_train=r2_score(train[item],predict) 
        mae_train=mean_absolute_error(train[item],predict)
        mse_train=mean_squared_error(train[item],predict)
        result = pd.concat([result,predict],axis=1)

        print('r2 test='+str(r2_test)+',mae test='+str(mae_test)+',mse test='+str(mse_test),file=f)
        print('r2 train='+str(r2_train)+',mae train='+str(mae_train)+',mse test='+str(mse_train),file=f)

pd.concat([test,result],axis=1).to_csv('output/auto_result.csv',index=False)

