import scipy.io as sio
import numpy as np
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout
from keras.utils import np_utils

NUMBER_OF_FEATURES=4
NUMBER_OF_CLASSES=6

def loaddata(dir):
    dataset=sio.loadmat(dir)
    palm_data=(abs(dataset['palm_ch1'])+abs(dataset['palm_ch2']))/2.00000
    lat_data=(abs(dataset['lat_ch1'])+abs(dataset['lat_ch2']))/2.00000
    tip_data=(abs(dataset['tip_ch1'])+abs(dataset['tip_ch2']))/2.00000
    spher_data=(abs(dataset['spher_ch1'])+abs(dataset['spher_ch2']))/2.00000
    hook_data=(abs(dataset['hook_ch1'])+abs(dataset['hook_ch2']))/2.00000
    cyl_data=(abs(dataset['cyl_ch1'])+abs(dataset['cyl_ch2']))/2.00000
    #print cyl_data.shape
    return palm_data, lat_data, tip_data, spher_data, hook_data, cyl_data

def calc_IEMG(raw_data):
    temp=0
    IEMG=[]
    timesteps, length=raw_data.shape
    for i in range(timesteps):
        for j in range(length):
            temp=temp+raw_data[i,j]
        IEMG.append(temp)
        temp=0
    return IEMG

def calc_SSC(raw_data):
    temp=0
    SSC=[]
    timesteps, length=raw_data.shape
    for i in range(timesteps):
        for j in range(length-2):
            if((raw_data[i,j+1]<raw_data[i,j+2])and(raw_data[i,j+1]<raw_data[i,j]))or((raw_data[i,j+1]>raw_data[i,j+2])and(raw_data[i,j+1]>raw_data[i,j])):
                temp=temp+1
        SSC.append(temp)
        temp=0
    return SSC


def calc_CZ(raw_data):
    temp=0
    CZ=[]
    timesteps, length=raw_data.shape
    for i in range(timesteps):
        theta=0.025*(raw_data[i,:].max()-raw_data[i,:].min())
        for j in range(length-1):
            if(raw_data[i,j]>theta and raw_data[i,j+1]<theta)or(raw_data[i,j]<theta and raw_data[i,j+1]>theta):
                temp=temp+1
        CZ.append(temp)
        temp=0               

    return CZ

def calc_WL(raw_data):
    temp=0
    WL=[]
    timesteps, length=raw_data.shape
    for i in range(timesteps):
        for j in range(length-1):
            temp=temp+abs(raw_data[i,j+1]-raw_data[i,j])
        WL.append(temp)
        temp=0

    return WL

def extractfeatures(raw_data):
    timesteps=raw_data.shape[0]
    features=[]
    temp=[]
    IEMG=calc_IEMG(raw_data)
    CZ=calc_CZ(raw_data)
    SSC=calc_SSC(raw_data)
    WL=calc_WL(raw_data)
    for i in range(timesteps):
        temp.append(IEMG[i])
        temp.append(CZ[i])
        temp.append(SSC[i])
        temp.append(WL[i])
        features.append(temp)
        temp=[]

    return features

def combinfeatures(palm,lat,tip,hook,spher,cyl):
    #combin=[]
    y=[]
    combin=np.append(palm,lat,axis=0)
    combin=np.append(combin,tip,axis=0)
    combin=np.append(combin,hook,axis=0)
    combin=np.append(combin,spher,axis=0)
    combin=np.append(combin,cyl,axis=0)
    for i in range(np.shape(palm)[0]):
        y.append(0)
    for i in range(np.shape(lat)[0]):
        y.append(1)
    for i in range(np.shape(tip)[0]):
        y.append(2)
    for i in range(np.shape(hook)[0]):
        y.append(3)
    for i in range(np.shape(spher)[0]):
        y.append(4)
    for i in range(np.shape(cyl)[0]):
        y.append(5)
    return combin, y
    
    

def build_model(first_layer_neurons, second_layer_neurons,input_shape):
    model = Sequential()
    model.add(LSTM(first_layer_neurons, input_dim=NUMBER_OF_FEATURES, dropout_U=0.3))
    model.add(Dense(second_layer_neurons))
    model.add(Dropout(0.2))
    model.add(Dense(NUMBER_OF_CLASSES, activation="softmax"))
    #model.add(Dense(1, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model



def main():
    palm_data, lat_data, tip_data, spher_data, hook_data, cyl_data=loaddata('database1/female_3.mat')
    palm_features=extractfeatures(palm_data)
    lat_features=extractfeatures(lat_data)
    tip_features=extractfeatures(tip_data)
    hook_features=extractfeatures(hook_data)
    spher_features=extractfeatures(spher_data)
    cyl_features=extractfeatures(cyl_data)
    x, y=combinfeatures(palm_features,lat_features,tip_features,hook_features,spher_features,cyl_features)
    x=np.array(x).reshape((180,1,4))
    y=np_utils.to_categorical(y, nb_classes=6)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    input_shape=x_train.shape[1:]
    model = build_model(150, 100,input_shape)
    print(model.summary())
    model.fit(x_train, y_train, nb_epoch=200, batch_size=50, verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == '__main__':
    main()


    


