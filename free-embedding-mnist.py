import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X,y = mnist["data"], mnist["target"]
#shuffle_index = np.random.permutation(60000)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Defining the free embedding with respect to features. 
def split_free_embed(X,features):
    n= X.shape
    div = int(features / 4)
    k, l = int(np.floor(n[1] / div)), int(np.ceil(n[1]/div))
    a, b = np.array([[k,l],[1,1]]), np.array([n[1],div])
    x,y = np.linalg.solve(a,b)
    x,y = int(np.rint(x)), int(np.rint(y))
    Z = np.zeros((n[0],div,2,2))
    for row in range(n[0]):
        for i in range(x):
            Z[row,i] = np.identity(2)
            for j in range(k):
                if X[row,k*i+j] <= 127:
                    Z[row,i] = np.matmul(Z[row,i],[[1,2],[0,1]]) #embedding of free group
                else:
                    Z[row,i] = np.matmul(Z[row,i],[[1,0],[2,1]]) #embedding of free group
                
        for i in range(y):
            Z[row,x + i] = np.identity(2)
            for j in range(l):
                if X[row,x*k + l*i+j] <= 127:
                    Z[row,x+i] = np.matmul(Z[row,x+i],[[1,2],[0,1]])
                else:
                    Z[row,x+i] = np.matmul(Z[row,x+i],[[1,0],[2,1]])
    Y = Z.reshape(n[0],features)
    return Y


features = 500 #needs to be divisible by 4

X_train_split_fe = split_free_embed(X_train,features)
X_test_split_fe = split_free_embed(X_test,features)

#scaling
data = X_train_split_fe
scaler = StandardScaler()
scaler.fit(data)
X_scaled_train_split_fe = scaler.transform(data)

datatest = X_test_split_fe
scaler = StandardScaler()
scaler.fit(datatest)
X_scaled_test_split_fe = scaler.transform(datatest)

#layers in neural network
n_inputs = features
n_hidden1 = int(features / 2)
n_hidden2 = int(n_hidden1 / 2)
n_outputs = 10


#contruction of dnn
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope = "hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn = None)
    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(60000 // batch_size):
            shuffle_index = np.random.permutation(60000)
            batch_index = shuffle_index[0:batch_size]
            X_batch, y_batch = X_scaled_train_split_fe[batch_index], y_train[batch_index]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_scaled_test_split_fe,y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./f2_model_final.ckpt")

