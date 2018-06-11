import mysql.connector
from underthesea import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer

config = {
    'user': 'cuonggt',
    'password': 'LklNtiuj3qq0iNFsEmZQ',
    'host': '35.194.220.210',
    'database': 'price_management',
    'raise_on_warnings': True,
}

cnx = mysql.connector.connect(**config)

cursor = cnx.cursor(buffered=True)
sql = "SELECT * FROM products LIMIT  2"
cursor.execute(sql)
data = cursor.fetchall()
s1 = []
tex = [[row[3]] for row in data]

print(tex[0])

for i in range(0, len(tex)):
    vectorizer = CountVectorizer()
    vectorizer.fit(tex[i])
    tex[i] = vectorizer.get_feature_names()

# print(vectorizer.get_feature_names())

print(tex)

mlb = MultiLabelBinarizer()
vector = mlb.fit_transform(tex)
print(vector)
# print(mlb.classes_[1])
# print(len(mlb.classes_))
cnx.close()
