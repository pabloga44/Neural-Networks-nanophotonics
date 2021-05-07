# This Python file uses the following encoding: utf-8
import main as mn

pathdata='/tfg_pablo/data/data_transmission_2.dat'
pathsave='/tfg_pablo/pre-trained/activation'

l,x,y=mn.get_transmissiondata1(pathdata)
x_train,x_validation,x_test,y_train,y_validation,y_test=mn.split_data(x,y)
model,history=mn.train_model(x_train,x_validation,y_train,y_validation,optimizer='Adam',actfun='relu',lr=0.00003,num_epochs=50000)

mn.save_model(model,history,l,x_train,x_validation,x_test,y_train,y_validation,y_test,route=pathsave,key='relu')
