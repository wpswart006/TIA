from openpyxl import load_workbook
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation

data = []
output = []
ws = load_workbook('../SAPBA_merged_LMALAN MASTER_PIETER FAANS-STRESSED16.xlsx').active
for row in ws.iter_rows(min_row=2):
    if (type(row[11].value) == float):
        data.append((row[3].value,row[4].value,row[5].value))
        output.append(1) if row[11].value >  0.75 else  output.append(0)

x_train = np.array(data[:np.ceil(len(data)*0.9).astype('uint16')])
x_test = np.array(data[np.ceil(len(data)*0.9).astype('uint16'):])

y_train = np.array(output[:np.ceil(len(output)*0.9).astype('uint16')])
y_train = np_utils.to_categorical(y_train) 
y_test = np.array(output[np.ceil(len(output)*0.9).astype('uint16'):])
y_test = np_utils.to_categorical(y_test) 

scale = np.max(x_train)
x_train /= scale
x_test /= scale

mean = np.std(x_train)
x_train -= mean
x_test -= mean
    
input_dim = x_train.shape[1]
nb_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(10, input_dim=input_dim))
model.add(Activation('elu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print("Training...")
history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=100,
                    verbose=1,
                    validation_data=(x_test, y_test))
res = model.predict(x_test)
_res = np.zeros(y_test.shape)
for k in range(res.shape[0]):
    _res[k][np.argmax(res[k])] = 1
    
hit = 0
for k in range (_res.shape[0]):
    if not (y_test[k]-_res[k]).all():
        hit +=1
        
print("Accuracy: ", hit/_res.shape[0])
