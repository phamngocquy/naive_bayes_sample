from __future__ import division

import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer
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
vectorizer = CountVectorizer()
new_mlb = MultiLabelBinarizer()

sql = "SELECT * FROM products LIMIT 10"
cursor.execute(sql)
data = cursor.fetchall()

# data_1
tex = []
for row in data:
    tex.append(row[3])

X = vectorizer.fit_transform(tex)
# print(X.toarray())
# print(vectorizer.get_feature_names())
corpus = [
    'This is the first document quy quy quy quy quy',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
print(type(corpus))
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names())

Y = new_mlb.fit_transform(tex)
print(Y)
print(new_mlb.classes_)