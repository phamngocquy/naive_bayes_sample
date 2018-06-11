from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time
import mysql.connector
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(buffered=True)
sql = "SELECT  id FROM categories"
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
print(np.asarray(matrix_Y))
np.savetxt("categories_matrix.csv", matrix_Y, delimiter=",")
