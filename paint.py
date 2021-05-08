# Import libraries

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split

import numpy as np # get the "numpy" library for linear algebra
import matplotlib.pyplot as plt # for plotting

def paint_spectra(l,x,y):
    # Plot some spectra from a set
    # l: wavelegnths
    # x: parameter (np.size(x) is the number os spectra in the set)
    # y: transmissions

    plt.figure(1)
    plt.plot(l, y[0,:], color='black', label='d = '+str(x[0])+' nm')
    plt.plot(l, y[int(np.size(x)/5.),:], color='red', label='d = '+str(x[int(np.size(x)/5.)])+' nm')
    plt.plot(l, y[int(np.size(x)/5.*2),:], color='green', label='d = '+str(x[int(np.size(x)/5.*2)])+' nm')
    plt.plot(l, y[int(np.size(x)/5.*3),:], color='cyan', label='d = '+str(x[int(np.size(x)/5.*3)])+' nm')
    plt.plot(l, y[int(np.size(x)/5.*4),:], color='orange', label='d = '+str(x[int(np.size(x)/5.*4)])+' nm')
    plt.plot(l, y[np.size(x)-1,:], color='blue', label='d = '+str(x[np.size(x)-1])+' nm')

    plt.yscale("log")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.xlabel("$\lambda\ [nm]$", fontsize=14)
    plt.ylabel("Transmission", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12) 

    plt.show()

    plt.figure(2)
    plt.plot(l, y[0,:], color='black', label='d = '+str(x[0])+' nm')
    plt.plot(l, y[int(np.size(x)/5.),:], color='red', label='d = '+str(x[int(np.size(x)/5.)])+' nm')
    plt.plot(l, y[int(np.size(x)/5.*2),:], color='green', label='d = '+str(x[int(np.size(x)/5.*2)])+' nm')
    plt.plot(l, y[int(np.size(x)/5.*3),:], color='cyan', label='d = '+str(x[int(np.size(x)/5.*3)])+' nm')
    plt.plot(l, y[int(np.size(x)/5.*4),:], color='orange', label='d = '+str(x[int(np.size(x)/5.*4)])+' nm')
    plt.plot(l, y[np.size(x)-1,:], color='blue', label='d = '+str(x[np.size(x)-1])+' nm')

    # plt.yscale("log")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.xlabel("$\lambda\ [nm]$", fontsize=14)
    plt.ylabel("Transmission", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12) 

    plt.show()


def paint_costs(history,save=False,route='',key=''):
    
    if type(history) is not dict:
        history=history.history

    # Cost function history
    costs = history['loss']
    val_costs = history['val_loss']

    # Plot the evoluation of the cost function
    plt.clf()
    plt.plot(costs)
    plt.plot(val_costs)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    if save:
        plt.savefig(str(route)+'/cost'+str(key)+'.pdf')
    plt.show()

def paint_predictions(model,l,x,y,save=False,route='',key=''):
    # Predict the spectra
    input = np.array([x[0]])
    output1 = model.predict(input)
    input = np.array([x[int(np.size(x)/5.)]])
    output2 = model.predict(input)
    input = np.array([x[int(np.size(x)/5.*2)]])
    output3 = model.predict(input)
    input = np.array([x[int(np.size(x)/5.*3)]])
    output4 = model.predict(input)
    input = np.array([x[int(np.size(x)/5.*4)]])
    output5 = model.predict(input)
    input = np.array([x[np.size(x)-1]])
    output6 = model.predict(input)

    plt.clf()

    # Plot the fits
    plt.figure(1)
    plt.plot(l, y[0,:], color='black', label='d = '+str(x[0])+' nm')
    plt.plot(l, 10**(output1.T), color='black', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.),:], color='red', label='d = '+str(x[int(np.size(x)/5.)])+' nm')
    plt.plot(l, 10**(output2.T), color='red', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.*2),:], color='green', label='d = '+str(x[int(np.size(x)/5.*2)])+' nm')
    plt.plot(l, 10**(output3.T), color='green', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.*3),:], color='cyan', label='d = '+str(x[int(np.size(x)/5.*3)])+' nm')
    plt.plot(l, 10**(output4.T), color='cyan', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.*4),:], color='orange', label='d = '+str(x[int(np.size(x)/5.*4)])+' nm')
    plt.plot(l, 10**(output5.T), color='orange', linestyle='--', label='fit')
    plt.plot(l, y[np.size(x)-1,:], color='blue', label='d = '+str(x[np.size(x)-1])+' nm')
    plt.plot(l, 10**(output6.T), color='blue', linestyle='--', label='fit')

    plt.yscale("log")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.xlabel("$\lambda\ [nm]$", fontsize=14)
    plt.ylabel("Transmission", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    if save:
        plt.savefig(str(route)+'/train'+str(key)+'_log.pdf')

    plt.show()

    plt.figure(2)
    plt.plot(l, y[0,:], color='black', label='d = '+str(x[0])+' nm')
    plt.plot(l, 10**(output1.T), color='black', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.),:], color='red', label='d = '+str(x[int(np.size(x)/5.)])+' nm')
    plt.plot(l, 10**(output2.T), color='red', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.*2),:], color='green', label='d = '+str(x[int(np.size(x)/5.*2)])+' nm')
    plt.plot(l, 10**(output3.T), color='green', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.*3),:], color='cyan', label='d = '+str(x[int(np.size(x)/5.*3)])+' nm')
    plt.plot(l, 10**(output4.T), color='cyan', linestyle='--', label='fit')
    plt.plot(l, y[int(np.size(x)/5.*4),:], color='orange', label='d = '+str(x[int(np.size(x)/5.*4)])+' nm')
    plt.plot(l, 10**(output5.T), color='orange', linestyle='--', label='fit')
    plt.plot(l, y[np.size(x)-1,:], color='blue', label='d = '+str(x[np.size(x)-1])+' nm')
    plt.plot(l, 10**(output6.T), color='blue', linestyle='--', label='fit')

    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.xlabel("$\lambda\ [nm]$", fontsize=14)
    plt.ylabel("Transmission", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    if save:
        plt.savefig(str(route)+'/train'+str(key)+'.pdf')

    plt.show()

def paint_training(routedata,keydata,save=False,route='',key=''):
    plt.clf()
    for i in keydata:
        history=np.load(routedata+'/history'+str(i)+'.npy',allow_pickle='TRUE').item()
        costs = history['loss']
        plt.plot(costs)
    plt.legend(keydata, loc='upper right')
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    if save:
        plt.savefig(str(route)+'/train'+str(key)+'.pdf')
    plt.show()