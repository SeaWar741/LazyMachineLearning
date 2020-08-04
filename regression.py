#no activation is needed, omit argument
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Input(shape=(1,)),
#    tf.keras.layers.Dense(1)
#])

#SGD stochastic gradient descent, iterative method for optimizing an objective function
#good for high dimensional optimization problems

#model.compile(optimizer = tf.keras.optimizer.SGD(0.001,0.9),loss='mse')

# reduce learning rate depending on the epoch number, learning rate scheduling

#loss
#MSE mean squared error, error cuadratico medio

#in regretion there is no accuracy,  we use R squared

#moores law, numbers of transitions sqinch on intregrated circuits doubles every two years

#c = Ar to t
#change it with log, make it linear

#logc = logr*t+logA

import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('moore.csv',header=None).values
X = data[:,0].reshape(-1,1) #to 2D array
Y = data[:,1]

#plot data
plt.scatter(X,Y)
plt.show()
plt.close()

#apply log to normalize to linear behaviour
Y = np.log(Y)
plt.scatter(X,Y)
plt.show()
plt.close()

#center x
X = X-X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

#function first, epch, learning rate
model.compile(optimizer=tf.keras.optimizers.SGD(0.001,0.9),loss='mse')
#model.compile(optimmizer='adam',loss='mse')

def schedule(epoch,lr):
    if epoch >= 50:
        return 0.0001
    return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule) 

#Train model
r = model.fit(X,Y, epochs=200,callbacks=[scheduler])

#plot loss
plt.plot(r.history['loss'], label = 'loss')
plt.show()
plt.close()

#get slope
print(model.layers) #all after input layers
print(model.layers[0].get_weights())
print(model.layers[0].get_weights()[0][0,0])
a=model.layers[0].get_weights()[0][0,0]

print("Time to double",np.log(2)/a)

#Analytical solution, faster, better. Its the exception on models. not the rule
X = np.array(X).flatten()
Y = np.array(Y)
denominator = X.dot(X) -X.mean()*X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denominator
b = (Y.mean() * X.dot(X) - X.mean()*X.dot(Y)/denominator)

print(a,b)
print("Time to double", np.log(2)/a)