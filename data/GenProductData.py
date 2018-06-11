import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer
from unidecode import unidecode

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}
cnx = mysql.connector.connect(**config)

cursor = cnx.cursor(buffered=True)
sql = "SELECT * FROM products LIMIT 1000"
cursor.execute(sql)
data = cursor.fetchall()

# data_1
# tex = [[row[3]] for row in data]
tex = []
for row in data:
    tex.append([row[3]])
for i in range(0, len(tex)):
    vectorizer = CountVectorizer()
    vectorizer.fit(tex[i])
    tex[i] = vectorizer.get_feature_names()

mlb = MultiLabelBinarizer()
v = mlb.fit_transform(tex)
print(v)
print(mlb.classes_)
print(len(mlb.classes_))
np.savetxt("product_matrix.csv", v, delimiter=",")
cnx.close()
