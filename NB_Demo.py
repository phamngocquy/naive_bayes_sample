import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer
from unidecode import unidecode
from sklearn.externals import joblib
from pathlib import Path


def load_data(filepath):
    my_clf_file = Path("train_data.pkl")
    if my_clf_file.is_file():
        clf = joblib.load("train_data.pkl")
    else:
        print("file not exit")


config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}
print("training...")
cnx = mysql.connector.connect(**config)

cursor = cnx.cursor(buffered=True)
sql = "SELECT * FROM products"
cursor.execute(sql)
data = cursor.fetchall()

# data_1
# tex = [[row[3]] for row in data]
tex = []
for row in data:
    tex.append([row[3]])
    tex.append([unidecode(row[3])])
for i in range(0, len(tex)):
    vectorizer = CountVectorizer()
    vectorizer.fit(tex[i])
    tex[i] = vectorizer.get_feature_names()

mlb = MultiLabelBinarizer()
v = mlb.fit_transform(tex)
# print(v)
# print(mlb.classes_)
# print(len(mlb.classes_))
# np.savetxt("product_matrix.csv", v, delimiter=",") // warning

# data2
s1 = []
for (row) in data:
    s1.append(row[12])
    s1.append(row[12])

clf = GaussianNB()
clf.fit(v, np.array(s1))

joblib.dump(clf, "train_data.pkl")

print("training complete")
while True:
    # input
    s = input("Enter something: ")
    input_ = [s]
    vectorizer_ = CountVectorizer()
    vectorizer_.fit(input_)
    list_input = list(vectorizer_.vocabulary_.keys())

    result_array = []
    check_correct = 0
    for i in range(0, len(mlb.classes_)):
        for j in range(0, len(list_input)):
            if mlb.classes_[i] == list_input[j]:
                result_array.append(1)
                check_correct = 1
                print("index: " + str(i + 1))
                print("true: " + mlb.classes_[i])
                break
        if check_correct == 0:
            result_array.append(0)
        else:
            check_correct = 0

    result_id = clf.predict([result_array])
    print(result_id)
    cursor.execute("SELECT name FROM categories WHERE  id = " + str(result_id[0]))
    final_data = cursor.fetchall()
    print(final_data)

cnx.close()
