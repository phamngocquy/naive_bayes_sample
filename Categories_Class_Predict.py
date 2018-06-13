from __future__ import division

import mysql.connector
import numpy as np
import tensorflow as tf
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
    sql = "SELECT  * FROM categories"
    cursor.execute(sql)
    data = cursor.fetchall()
    idX = []
    for id_categories in data:
        idX.append(id_categories[0])
    matrix_Y = np.array([idX])
    sql_get_category_id_by_product = "SELECT * FROM products WHERE  category_id != 9999"
    cursor.execute(sql_get_category_id_by_product)
    data_category_id_by_product = cursor.fetchall()

    for category_id in data_category_id_by_product:
        add_array = []
        for i_ in range(0, len(idX)):
            if category_id[12] == idX[i_]:
                add_array.append(1)
            else:
                add_array.append(0)
        matrix_Y = np.vstack([matrix_Y, add_array])
    matrix_Y = np.delete(matrix_Y, 0, axis=0)

    return matrix_Y, data


def loading_train_product_data():
    sql = "SELECT * FROM products WHERE  category_id != 9999 "
    cursor.execute(sql)
    data = cursor.fetchall()

    tex = []
    for row in data:
        tex.append(row[3])

    vectorizer = CountVectorizer()
    tmp = vectorizer.fit_transform(tex)
    v = tmp.toarray()
    return v, vectorizer.get_feature_names()


trainX_Product, mblClasse_ = loading_train_product_data()
trainY_Category, correct_Data_Category = load_train_category_data()

# GLOBAL PARAMETERS

numFeatures = trainX_Product.shape[1]
# numLabels = number of classes we are predicting (here just 2: ham or spam)
numLabels = trainY_Category.shape[1]
# create a tensorflow session
sess = tf.Session()

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

weights = tf.Variable(tf.zeros([numFeatures, numLabels]))

bias = tf.Variable(tf.zeros([1, numLabels]))

# OPS / OPERATIONS
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
init_OP = tf.initialize_all_variables()
sess.run(init_OP)
saver = tf.train.Saver()
saver.restore(sess, "trained_variables.ckpt")
print("restore success")

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

    testX = np.array(result_array)
    print(testX)
    tensor_prediction = sess.run(activation_OP, feed_dict={X: testX.reshape(1, len(testX))})
    print(tensor_prediction)
    print("result: ")

    for i in range(0, len(tensor_prediction[0])):
        if tensor_prediction[0][i] == max(tensor_prediction[0]):
            print(i)
            print(tensor_prediction[0][i])
            print(correct_Data_Category[i][6])

    cnx.close()
