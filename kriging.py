#%%
import numpy as np
from pyKriging.krige import kriging
import pandas as pd
import pickle as pkl

#%%

df_train = pd.read_csv('dis_23.csv')
input_Act_labels = ['A14','A23']
input_T_labels = ['Tup','Tmid','Tdown']
test_Act =  (df_train[input_Act_labels]-0.09)/0.01
test_T =  df_train[input_T_labels]/150

x_test=np.append(test_Act, test_T, axis=1)


#iii = ['200','300','400','500','600','700','800']
iii = ['600']


for data in iii:

    with open(f'kriging_{data}.pkl','rb') as f:
        k_trained = pkl.load(f)

    k_pred = [k_trained.predict(x_) for x_ in x_test]

    FEM_23z =  df_train['23z']

    k_pred_mis=abs(k_pred-FEM_23z)

    k_error=k_pred_mis/FEM_23z



    mean_error=np.mean(k_error)

    mean_error

    np.savetxt(f"predict_{data}.csv",k_pred,delimiter=',')


# %%
df_train = pd.read_excel('C:/Users/DL/Desktop/新出图/fig14/1000/y_9to10_150_5inp_kriging.xlsx')
input_Act_labels = ['A14','A23']
input_T_labels = ['Tup']
test_Act =  (df_train[input_Act_labels]-0.05)/0.05
test_T =  df_train[input_T_labels]/150

x_test=np.column_stack((test_Act,test_T))
#x_test=np.append(test_Act, test_T, axis=1)

# %%a
with open(f'C:/Users/DL/Desktop/新出图/fig14/1000/y_9to10_150_5inp_kriging.pkl','rb') as f:
    k_trained = pkl.load(f)

#%%
%% time
k_pred = [k_trained.predict(x_) for x_ in x_test]

np.savetxt(f"kriging查看离散性的对角线_y.csv",k_pred,delimiter=',')


























#%%
df_train=pd.read_excel('H:/新建文件夹 (2)/9/dis_23.xlsx')  

input_Act_labels = ['A14','A23']
input_T_labels = ['Tup','Tmid','Tdown']
test_Act =  (df_train[input_Act_labels]-0.09)/0.01
test_T =  df_train[input_T_labels]/150

x_test=np.append(test_Act, test_T, axis=1)

#公式     x1_norm = (x1 - x1.min()) / (x1.max() - x1.min()) 

#x_test  =pd.concat([x11, x12], axis=1)
#x_predict=x11

# %%a
with open(f'C:/Users/DL/Desktop/新出图/fig8/新的包含不收敛的训练/新9-10/y_9to10_150_5inp_kriging.pkl','rb') as f:
    k_trained = pkl.load(f)

%time   k_pred = [k_trained.predict(x_) for x_ in x_test]

np.savetxt(f"H:/新建文件夹 (2)/9/kriging_y.csv",k_pred,delimiter=',')

#%%
    FEM_23z =  df_train['23y']

    k_pred_mis=abs(k_pred-FEM_23z)

    k_error=k_pred_mis/FEM_23z



    mean_error=np.mean(k_error)

    mean_error

# %%

