import tensorflow as tf

#load data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print(type(data))

#sklearn.utils.Bunch

print(data.keys())

#check data size
print(data.data.shape)

print(data.target)

print(data.target_names)

#input names, names of each feature
print(data.target.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data.data,data.target,test_size=0.33)
N, D = X_train.shape

#scale data
from sklearn.preprocessing import StandardScaler

#normalization mean divided by standard deviation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Tensorflow -->
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

#otro método
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(1,input_shape=(D,),activation='sigmoid')

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

#train model
r = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=100)

#evaluate model
print("Train Score:",model.evaluate(X_train,Y_train))
print("Test Score:",model.evaluate(X_test,Y_test))

#plotting fit
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label = 'val_loss')
plt.legend()
plt.show()

#plot accuracy
plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()