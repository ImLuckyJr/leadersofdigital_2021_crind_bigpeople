import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import requests
import pandas as pd
import datetime
import json
import numpy as np
import math as m
import sys
from io import StringIO
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, RepeatVector, Flatten, merge
import datetime
from keras.models import model_from_json
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU, TimeDistributed
from keras.layers.recurrent import LSTM
from sklearn import preprocessing
from keras.callbacks import EarlyStopping

from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import average
from keras.layers import Input, Reshape, Dropout, SpatialDropout1D
from keras.models import Model
from keras.optimizers import SGD, Adam

def create_mlp_attention(load = True, num=0, name = 'best_mlp_attention'):
    # create model
    if load:
        json_file = open('./models/' + name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights('./models/' + name + '.h5')
        print("Loaded model from disk")
        return model
    # load json and create model
    else:
        D = 0.2
        S = 1
        ### INPUT DATA
        inputs = Input(shape=(num,))
        ### DEFINE A MULTILAYER PERCEPTRON NETWORK
        attention_probs = Dense(num, activation='softmax')(inputs)
        attention_mul = merge([inputs, attention_probs], output_shape=32, mode='mul')
        mlp_net = Dense(64, activation='relu', kernel_initializer='he_uniform')(attention_mul)
        mlp_net = Dropout(rate=0.2)(mlp_net)
        mlp_out = Dense(1, activation='linear')(mlp_net)
        att_mdl = Model(inputs=inputs, outputs=mlp_out)
        return att_mdl

def create_mlp_cnn(load = True, num=0, name = 'best_mlp_cnn'):
    # create model
    if load:
        json_file = open('./models/' + name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights('./models/' + name + '.h5')
        print("Loaded model from disk")
        return model
    # load json and create model
    else:
        D = 0.2
        S = 1
        ### INPUT DATA
        inputs = Input(shape=(num,))
        ### DEFINE A MULTILAYER PERCEPTRON NETWORK
        mlp_net = Dense(64, activation='relu', kernel_initializer='he_uniform')(inputs)
        mlp_net = Dropout(rate=D, seed=S)(mlp_net)
        mlp_net = Dense(64, activation='relu', kernel_initializer='he_uniform')(mlp_net)
        mlp_net = Dropout(rate=D, seed=S)(mlp_net)
        mlp_out = Dense(1, activation='sigmoid')(mlp_net)
        mlp_mdl = Model(inputs=inputs, outputs=mlp_out)
        ### DEFINE A CONVOLUTIONAL NETWORK
        cnv_net = Reshape((X.shape[1], 1))(inputs)
        cnv_net = Conv1D(32, 4, activation='relu', padding="same", kernel_initializer='he_uniform')(cnv_net)
        cnv_net = MaxPooling1D(2)(cnv_net)
        cnv_net = SpatialDropout1D(D)(cnv_net)
        cnv_net = Flatten()(cnv_net)
        cnv_out = Dense(1, activation='relu')(cnv_net)
        cnv_mdl = Model(inputs=inputs, outputs=cnv_out)
        ### COMBINE MLP AND CNV
        con_out = average([mlp_out, cnv_out])
        con_mdl = Model(inputs=inputs, outputs=con_out)
        return con_mdl

def create_my_mlp(load = True, num=0, name = 'best_mlp'):
    # create model
    if load:
        json_file = open('./models/'+name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights('./models/'+name+'.h5')
        print("Loaded model from disk")
        return model
    # load json and create model
    else:
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape = (num,)))
        model.add(Dense(64, activation='relu'))
        ##model.add(LeakyReLU())
        model.add(Dense(1))
    ##model.summary()
    return model

def save_model(model, name):
    model_json = model.to_json()
    with open("./models/"+name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./models/"+name+".h5")
    print("Saved model to disk")

def visual_history(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


def getData(X,Y, XY, df):
    pr_orig = ['year', 'month', 'day', 'hour', 'dayofweek', 'max_ots', 'is_holiday', 'is_before_holiday']
    pr = ['month', 'day', 'hour', 'dayofweek', 'is_holiday', 'is_before_holiday', '1_h_ago','2_h_ago','3_h_ago','4_h_ago','5_h_ago','6_h_ago','7_h_ago','8_h_ago','9_h_ago','10_h_ago','11_h_ago','12_h_ago','13_h_ago','14_h_ago','15_h_ago','16_h_ago','17_h_ago','18_h_ago','19_h_ago','20_h_ago','21_h_ago','22_h_ago','23_h_ago','24_h_ago']
    lpr = len(pr)
    for index, row in df.iterrows():
        for i in range(lpr):
            X[index][i] = row[pr[i]]
            XY[index][i] = row[pr[i]]
        # X[index][0] = row['year']
        # X[index][1] = row['month']
        # X[index][2] = row['day']
        # X[index][3] = row['hour']
        # X[index][4] = row['dayofweek']
        # X[index][5] = row['max_ots']
        # X[index][6] = row['is_holiday']
        # X[index][7] = row['is_before_holiday']
        Y[index][0] = row['Mac']
        XY[index][lpr] = row['Mac']
    return lpr

try:
    df = pd.read_csv('csv_holidays_last_24_hours.csv', sep=',')
    property = len(df.columns) - 1
    count = len(df.index)
    ##df.plot(kind='bar', x='AddedOnDate', y='Mac')
    ##plt.show()
    sizeXY = (count, property + 1)
    XY = np.ones(sizeXY)
    sizeX = (count, property)
    X = np.ones(sizeX)
    sizeY = (count, 1)
    Y = np.ones(sizeY)
    lpr = getData(X, Y, XY, df)
    scale = True
    if scale:
        ##scaler = preprocessing.MinMaxScaler()
        ##scaler.fit(XY)
        ##XY_scale = scaler.transform(XY)
        for j in range(count):
            for l in range(lpr):
                X[j][l] = (XY[j][l] - XY.min())/(XY.max()-XY.min())
            Y[j][0] = (XY[j][lpr] - XY.min())/(XY.max()-XY.min())
    test = int(X.shape[0] * 0.7)
    X_train = X[:test]
    Y_train = Y[:test]
    X_test = X[test:]
    Y_test = Y[test:]
    load = False
    name = 'create_my_mlp'
    model = create_my_mlp(load, property, name)
    epochs = 300
    erl_stop = EarlyStopping(monitor='val_loss', patience=100)
    ##optimizer='adam'
    adam = Adam(lr=1.2)

    ##name = 'create_mlp_cnn'
    ##model = create_mlp_cnn(load, property, name)
    ##epochs = 10000

    ##name = 'create_mlp_att'
    ##model = create_mlp_attention(load, property, 'create_mlp_att')
    print('run')
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print('compile ok')
    if load != True:
        history = model.fit(X_train, Y_train, epochs=100, verbose=1, batch_size=24, validation_data=(X_test, Y_test))
        visual_history(history)
        save_model(model, name)
    else:
        ##X_test = np.array([[ 21.0,   22.0,   11.5]])
        Y_predict = model.predict(X_test)
        # show the inputs and predicted outputs
        y = []
        x_r = []
        x_p = []
        for i in range(len(X_test)):
            if scale:
                print("Real=%s, Predicted=%s" % ((Y_test[i]*(XY.max()-XY.min())+XY.min()), (Y_predict[i][0]*(XY.max()-XY.min())+XY.min())))
                x_r.append((Y_test[i]*(XY.max()-XY.min())+XY.min()))
                x_p.append((Y_predict[i][0]*(XY.max()-XY.min())+XY.min()))
                y.append(i)
            else:
                print("Real=%s, Predicted=%s" % (Y_test[i], Y_predict[i][0]))
                x_r.append(Y_test[i])
                x_p.append(Y_predict[i][0])
                y.append(i)
        plt.figure()
        plt.plot(y, x_r, 'r', label="Real")
        plt.plot(y, x_p, 'b', label="Predicted")
        plt.legend(loc="upper left")
        plt.title('Eq model')
        plt.ylabel('value')
        plt.xlabel('index')
        plt.show()
except Exception as e:
    print(e)