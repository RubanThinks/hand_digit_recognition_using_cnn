

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)= keras.datasets.mnist.load_data()

x_train

x_train=x_train/255
x_test=x_test/255

x_train.shape

plt.imshow(x_train[356])



x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)

x_train.shape

x_test.shape

x_train_flattened.shape

model= keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3),activation="relu",input_shape=(28,28,1)),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(32, kernel_size=(3,3),activation="relu"),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation="sigmoid"),
    keras.layers.Dense(10,activation="softmax")])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10)

model.evaluate(x_test,y_test)

y_pred=model.predict(x_test)

plt.matshow(x_test[9])

y_pred[9]

np.argmax(y_pred[9])

y_pred_labels=[np.argmax(i)for i in y_pred]

cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)

import seaborn as sn
plt.figure(figsize=(10,10))
sn.heatmap(cm,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")

model.save("final.h5")
