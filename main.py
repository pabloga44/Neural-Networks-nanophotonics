# Import libraries

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.model_selection import train_test_split

import numpy as np # get the "numpy" library for linear algebra
import matplotlib.pyplot as plt # for plotting

def get_transmissiondata1(file,lnum=401,dnum=400,stepl=2,steps=4):
    # Collects and arrange data from transmission spectra WHEN THERE IS ONLY ONE PARAMETER.
    #   file: file that contains data
    #   lnum: number of points in the spectrum
    #   dnum: number of sprectra
    #   stepl: step to choose wavelength
    #   steps: step to choose the number of spectres in the data set

    out=np.loadtxt(str(file),dtype='float',delimiter=',')

    xx=out[:,0] # Longitudes de onda
    xx=np.reshape(xx,(dnum,lnum))
    yy=out[:,1] # Periodicidadess
    yy=np.reshape(yy,(dnum,lnum))
    zz=out[:,2] # Transmisiones
    zz=np.reshape(zz,(dnum,lnum))

    l=xx[0,1:-1:stepl] # Wavelengths
    x=yy[0::steps,0] # Periodicities
    y=zz[0::steps,1:-1:stepl] # Transmissions

    return l,x,y

def get_transmissiondata():
    # Aquí va una función para el nuevo tipo de archivos de datos cuando haya más de un parámetro
    return 1

def split_data(x,y,validationsize=0.1,testsize=0.1):
    # Splits data in train, validation and test sets
    #   x: input data
    #   y: output data
    #   validationsize: size of validation set (up to 1)
    #   testsize: size of test set (up to 1)

    x_train, x_2, y_train, y_2 = train_test_split(x, y, test_size=validationsize+testsize)
    x_validation, x_test, y_validation, y_test = train_test_split(x_2, y_2, test_size=testsize/(validationsize+testsize))

    idx=np.argsort(x_train)
    idx2=np.argsort(x_validation)
    idx3=np.argsort(x_test)

    x_train=np.array(x_train)[idx]
    x_validation=np.array(x_validation)[idx2]
    x_test=np.array(x_test)[idx3]
    y_train=np.array(y_train)[idx]
    y_validation=np.array(y_validation)[idx2]
    y_test=np.array(y_test)[idx3]

    return x_train,x_validation,x_test,y_train,y_validation,y_test

def train_model(x_train,x_validation,y_train,y_validation,optimizer='Adam',actfun='sigmoid',lr=0.0006,num_epochs=50000,verb=0,hidden=3,neurons=150,batch=80):
    # Nos permite seleccionar: 
    #   función activación, optimizador, learning rate, num epochs, hidden layes, num neurons/hid layer, batch size y si queremos verbose.
    
    # Build the model
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation=actfun, input_shape=(1,)))
    for h in range(hidden-1): #Siempre hay como mínimo una hidden layer
      model.add(layers.Dense(neurons, activation=actfun))
    model.add(layers.Dense(8))
    model.summary()

    # Compile the model
    if optimizer=='Adam':
        opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer=='SGD':
        opt = keras.optimizers.SGD(learning_rate=lr)
    elif optimizer=='Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate=lr)
    elif optimizer=='RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer=='Nadam':
        opt = keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch, validation_data = (x_validation,y_validation), verbose=verb)

    return model,history

def error_model(model,x_train,x_validation,x_test,y_train,y_validation,y_test):

    print(model.evaluate(x_train,y_train))
    print(model.evaluate(x_validation,y_validation))
    print(model.evaluate(x_test,y_test))

def save_model(model,history,l,x_train,x_validation,x_test,y_train,y_validation,y_test,route='',key='',):
    # Save the trained model and its training set
    #   route: folder where to save the model
    #   key: a key word to identify this particular model

    # route='results/'+str(route)

    if type(history) is not dict:
        history=history.history

    model.save(route+'/model'+str(key)+'.h5')
    np.save(route+'/history'+str(key)+'.npy',history)

    np.save(route+'/wavelength'+str(key)+'.npy',l)

    np.save(route+'/x_train'+str(key)+'.npy',x_train)
    np.save(route+'/x_validation'+str(key)+'.npy',x_validation)
    np.save(route+'/x_test'+str(key)+'.npy',x_test)

    np.save(route+'/y_train'+str(key)+'.npy',y_train)
    np.save(route+'/y_validation'+str(key)+'.npy',y_validation)
    np.save(route+'/y_test'+str(key)+'.npy',y_test)

    np.save(route+'/y_test'+str(key)+'.npy',y_test)

    print('Model saved correctly!')

def load_model(route='',key=''):
    # Load a previously saved model

    # route='results/'+str(route)

    model=keras.models.load_model(route+'/model'+str(key)+'.h5')
    history=np.load(route+'/history'+str(key)+'.npy',allow_pickle='TRUE').item()

    x_train=np.load(route+'/x_train'+str(key)+'.npy')
    x_validation=np.load(route+'/x_validation'+str(key)+'.npy')
    x_test=np.load(route+'/x_test'+str(key)+'.npy')

    y_train=np.load(route+'/y_train'+str(key)+'.npy')
    y_validation=np.load(route+'/y_validation'+str(key)+'.npy')
    y_test=np.load(route+'/y_test'+str(key)+'.npy')

    try:
        l=np.load(route+'/wavelength'+str(key)+'.npy')
    except:
        out=np.loadtxt('data/data_transmission_2.dat',dtype='float',delimiter=',')
        xx=out[:,0]
        xx=np.reshape(xx,(400,401))
        l=xx[0,1:-1:2]
    
    if len(l)==len(y_train[0,:]):
        return model,history,l,x_train,x_validation,x_test,y_train,y_validation,y_test
    else:
        print('WARNING! No wavelengths found!')
        return model,history,False,x_train,x_validation,x_test,y_train,y_validation,y_test
    
#Por Pablo    
def clean_data(x,y):
    # Elimina las filas de NaN y te dice cuales son los parámetros que han dado error
    nans = [];
    for index in range(len(y)):

      if False in (y[index]==y[index]): #La forma de encontrar un NaN es porque (NaN==NaN)=>False.
        nans.append(index)
        print('These params caused NaN, so they were removed:',x[index])

    for i in reversed(nans):
      x = np.delete(x,i,axis=0)
      y = np.delete(y,i,axis=0)

    return x,y

def get_data(file_l,file_x,file_y, clean=True):
  # Función que recibe los archivos de texto y devuelve los datos en ndarrays
  # Independiente del número de parámetros
  # Puede devolverlos ya limpios
  l = np.loadtxt(str(file_l),dtype=float)
  x = np.loadtxt(str(file_x),dtype=float)
  y = np.loadtxt(str(file_y),dtype=float)

  if clean:
    x,y = clean_data(x,y)
  
  return l,x,y

def get_data_1param(file_l,file_x,file_y, clean=True):
  # Función que recibe los archivos de texto y devuelve los datos en ndarrays
  # Independiente del número de parámetros
  # Puede devolverlos ya limpios
  l = np.loadtxt(str(file_l),dtype=float)
  In = np.loadtxt(str(file_x),dtype=float)
  x = In[:,2]
  y = np.loadtxt(str(file_y),dtype=float)

  if clean:
    x,y = clean_data(x,y)
  
  return l,x,y
