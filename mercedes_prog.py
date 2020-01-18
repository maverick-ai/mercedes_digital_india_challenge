import numpy as np
import pandas as pd
from keras import layers
from keras import Input
from keras.models import Model
from keras.utils.np_utils import to_categorical
from collections import Counter
print("The data should contain the last two coloumns as labels and should be in csv format")
print("*** The second last coloumn should be the car bought by the customer and the last coloumn should be the ideal financial option chosen by the customer ***")
data_address=str(input("Enter the location of data for Training :"))
header_info=str(input("Does data conatins header? [y/n] :"))
if header_info=="y":
    data=pd.read_csv(data_address)
else:
    data=pd.read_csv(data_address,header=None)
data=np.array(data,dtype='float32')
label_car_model=data[:,-2]
label_finance_option=data[:,-1]
data=data[:,:-2]
num_of_car_models=len(Counter(label_car_model[0:]).keys())
num_of_finance_options=len(Counter(label_finance_option[0:]).keys())
label_car_model=to_categorical(label_car_model)
label_finance_option=to_categorical(label_finance_option)
x1=Input(shape=(None,),dtype='float32',name='data_input')
x=layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu')(x1)
x=layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu')(x)
x=layers.Dense(64,activation='relu')(x)
x=layers.Dense(32,activation='relu')(x)
x=layers.Dense(32,kernel_regularizer=regularizers.l2(0.001),activation='relu')(x)
x=layers.Dense(16,activation='relu')(x)
finance_option_prediction=layers.Dense(num_of_finance_options,activation='softmax',name='fin_pred')(x)
car_model_prediction=layers.Dense(num_of_car_models,activation='softmax',name='car_pred')(x)
model=Model(x1,[finance_option_prediction,car_model_prediction])
model.compile(optimizer='rmsprop',loss={'fin_pred':'categorical_crossentropy','car_pred':'categorical_crossentropy'})
model.fit(data,[label_finance_option,label_car_model],epochs=10,batch=32)
