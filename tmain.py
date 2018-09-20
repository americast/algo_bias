import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import keras
import pudb

EPOCHS = 100

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h_1 = tf.nn.selu((tf.matmul(X, w_1)))  # The \sigma function
    h_2 = tf.nn.selu((tf.matmul(h_1, w_2)))  # The \sigma function
    h_3 = tf.nn.selu((tf.matmul(h_2, w_3)))  # The \sigma function
    h_4 = tf.nn.selu((tf.matmul(h_3, w_4)))  # The \sigma function
    h_5 = tf.nn.selu((tf.matmul(h_4, w_5)))  # The \sigma function
    h_6 = tf.nn.selu((tf.matmul(h_5, w_6)))  # The \sigma function
    h_7 = tf.nn.selu((tf.matmul(h_6, w_7)))  # The \sigma function
    h_8 = tf.nn.selu((tf.matmul(h_7, w_8)))  # The \sigma function
    yhat = tf.nn.selu((tf.matmul(h_8, w_9)))  # The \sigma function
    
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def main():
    global EPOCHS
    # train_X, test_X, train_y, test_y = get_iris_data()

    # Saver
    name = ""

    print("Train? (y for train, n for test)")
    choice = raw_input()
    train_flag = True
    if (choice =='n' or choice=='N'):
          df = pd.read_csv("data/out-test.csv")
          BATCH_SIZE = df.shape[0]
          EPOCHS = 1
          train_flag = False
          name = raw_input("Enter model file name: ")
    else:
         df = pd.read_csv("data/out-train.csv")



    cols = df.columns.values
    cols = np.delete(cols, [1])
    train_X = df.loc[:,cols].values

    train_y = df["decile_score"].values
    y_train_ = train_y
    train_y = keras.utils.np_utils.to_categorical(train_y)



    print train_X.shape
    print train_y.shape
    # exit()
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size_1 = 256                                # Number of hidden nodes
    h_size_2 = 256                                # Number of hidden nodes
    h_size_3 = 128                                # Number of hidden nodes
    h_size_4 = 64                                  # Number of hidden nodes
    h_size_5 = 64                                  # Number of hidden nodes
    h_size_6 = 32                                  # Number of hidden nodes
    h_size_7 = 16                                  # Number of hidden nodes
    h_size_8 = 8                                  # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size_1))
    w_2 = init_weights((h_size_1, h_size_2))
    w_3 = init_weights((h_size_2, h_size_3))
    w_4 = init_weights((h_size_3, h_size_4))
    w_5 = init_weights((h_size_4, h_size_5))
    w_6 = init_weights((h_size_5, h_size_6))
    w_7 = init_weights((h_size_6, h_size_7))
    w_8 = init_weights((h_size_7, h_size_8))
    w_9 = init_weights((h_size_8, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    saver = tf.train.Saver()
    # Run SGD
    sess = tf.Session()
    if not train_flag:
        saver.restore(sess, "checkpoints/"+name)

    if train_flag:
        init = tf.global_variables_initializer()
        sess.run(init)

    for epoch in range(EPOCHS):
        # Train with each example
        if train_flag:
            for i in range(len(train_X)):
                sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 tf.run(feed_dict={X: train_X}))
        # test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
        #                          sess.run(predict, feed_dict={X: test_X, y: test_y}))
        pu.db
        print("Epoch = %d, train accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy))
        if train_flag:
            saver.save(sess, "checkpoints/model_epoch_"+str(epoch)+".ckpt")

    sess.close()

if __name__ == '__main__':
    main()
