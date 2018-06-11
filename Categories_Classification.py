from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MultiLabelBinarizer
import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(buffered=True)


def load_train_category_data():
    sql = "SELECT  id,name FROM categories"
    cursor.execute(sql)
    data = cursor.fetchall()
    idX = []
    for id_categories in data:
        idX.append(id_categories[0])
    matrix_Y = np.array([idX])

    sql_get_category_id_by_product = "SELECT category_id FROM products LIMIT 1000 "
    cursor.execute(sql_get_category_id_by_product)
    data_category_id_by_product = cursor.fetchall()
    for category_id in data_category_id_by_product:
        add_array = []
        for i in range(0, len(idX)):
            if category_id[0] == idX[i]:
                add_array.append(1)
            else:
                add_array.append(0)

        matrix_Y = np.vstack([matrix_Y, add_array])
    matrix_Y = np.delete(matrix_Y, 0, axis=0)
    return matrix_Y, data


def loading_train_product_data():
    sql = "SELECT * FROM products LIMIT 1000"
    cursor.execute(sql)
    data = cursor.fetchall()

    # data_1
    # tex = [[row[3]] for row in data]
    tex = []
    for row in data:
        tex.append([row[3]])
    for i_ in range(0, len(tex)):
        vectorizer = CountVectorizer()
        vectorizer.fit(tex[i_])
        tex[i_] = vectorizer.get_feature_names()

    mlb = MultiLabelBinarizer()
    v = mlb.fit_transform(tex)
    return v, mlb.classes_


trainX_Product, mblClasse_ = loading_train_product_data()
trainY_Category, correct_Data_Category = load_train_category_data()

# GLOBAL PARAMETERS
numFeatures = trainX_Product.shape[1]
numLabels = trainY_Category.shape[1]
# print(numFeatures)
# print(numLabels)
numEpochs = 2000
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=trainX_Product.shape[0],
                                          decay_rate=0.95,
                                          staircase=True)

X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])
# print(X)
# print(yGold)
# VARIABLES

weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6 / numFeatures +
                                                       numLabels + 1)),
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1, numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6 / numFeatures + numLabels + 1)),
                                    name="bias"))

print(weights)
print(bias)

######################
### PREDICTION OPS ###
######################

# INITIALIZE our weights and biases
init_OP = tf.global_variables_initializer()

# PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

#####################
### EVALUATION OP ###
#####################

# COST FUNCTION i.e. MEAN SQUARED ERROR
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")

#######################
### OPTIMIZATION OP ###
#######################

# OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

###########################
### GRAPH LIVE UPDATING ###
###########################

epoch_values = []
accuracy_values = []
cost_values = []
# Turn on interactive plotting
plt.ion()
# Create the main, super plot
fig = plt.figure()
# Create two subplots on their own axes and give titles
ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()

#####################
### RUN THE GRAPH ###
#####################

# Create a tensorflow session
sess = tf.Session()

# Initialize all tensorflow variables
sess.run(init_OP)

## Ops for vizualization

# ??????????????????
# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

# False is 0 and True is 1, what was our average?

accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)
# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
# Merge all summaries
all_summary_OPS = tf.summary.merge_all()
# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

# Initialize reporting variables
cost = 0
diff = 1

print("training")
# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence." % diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX_Product, yGold: trainY_Category})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP],
                feed_dict={X: trainX_Product, yGold: trainY_Category}
            )
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Write summary stats to writer
            writer.add_summary(summary_results, i)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            # generate print statements
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print("step %d, cost %g" % (i, newCost))
            print("step %d, change in cost %g" % (i, diff))

            # Plot progress to our two subplots
            accuracyLine, = ax1.plot(epoch_values, accuracy_values)
            costLine, = ax2.plot(epoch_values, cost_values)
            fig.canvas.draw()

print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                      feed_dict={X: trainX_Product,
                                                                 yGold: trainY_Category})))
##############################
### SAVE TRAINED VARIABLES ###
##############################

# Create Saver
saver = tf.train.Saver()
# Save variables to .ckpt file
saver.save(sess, "/home/haku/PycharmProjects/DemoOnce/trained_variables.ckpt")

print("train complete")

while True:
    s = input("Enter something: ")
    input_ = [s]
    vectorizer_ = CountVectorizer()
    vectorizer_.fit(input_)
    list_input = list(vectorizer_.vocabulary_.keys())

    result_array = []
    check_correct = 0
    for i in range(0, len(mblClasse_)):
        for j in range(0, len(list_input)):
            if mblClasse_[i] == list_input[j]:
                result_array.append(1)
                check_correct = 1
                print("index: " + str(i + 1))
                print("true: " + mblClasse_[i])
                break
        if check_correct == 0:
            result_array.append(0)
        else:
            check_correct = 0

    testX = np.array([result_array])
    prediction = sess.run(activation_OP, feed_dict={X: testX})

    print("result: ")
    print(correct_Data_Category[np.argmax(prediction[0])])

# How well do we perform on held-out test data?


############################
### MAKE NEW PREDICTIONS ###
############################

# Close tensorflow session
sess.close()

# To view tensorboard:
# 1. run: tensorboard --logdir=/path/to/log-directory
# 2. open your browser to http://localhost:6006/
# See tutorial here for graph visualization:
# https://www.tensorflow.org/versions/0.6.0/how_tos/graph_viz/index.html
cnx.close()
