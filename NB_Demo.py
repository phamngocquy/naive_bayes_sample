import mysql.connector
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer
from unidecode import unidecode
from sklearn.externals import joblib
from pathlib import Path
import pickle

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(buffered=True)


def predict_word(clf, mlb):
    while True:
        try:
            s = input("Enter something: ")
            input_ = [s]
            vectorizer_ = CountVectorizer()
            vectorizer_.fit(input_)
            list_input = list(vectorizer_.vocabulary_.keys())
            result_array = []
            check_correct = 0
            count_true = 0
            for i in range(0, len(mlb.classes_)):
                for j in range(0, len(list_input)):
                    if mlb.classes_[i] == list_input[j]:
                        result_array.append(1)
                        check_correct = 1
                        count_true = count_true + 1
                        print("index: " + str(i + 1))
                        print("true: " + mlb.classes_[i])
                        break
                if check_correct == 0:
                    result_array.append(0)
                else:
                    check_correct = 0
            if count_true > 0:
                result_id = clf.predict([result_array])
                print(result_id)
                cursor.execute("SELECT name FROM categories WHERE  id = " + str(result_id[0]))
                final_data = cursor.fetchall()
                print(final_data)
            else:
                print("haven't the correct word")
        except ValueError:
            print("invalid value")


def load_data(filepath, filepath_mlb):
    my_clf_file = Path(filepath)
    my_mlb_file = Path(filepath_mlb)
    if my_clf_file.is_file() and my_mlb_file.is_file():
        clf = pickle.load(open(filepath, 'rb'))
        mlb = pickle.load(open(filepath_mlb, 'rb'))
        predict_word(clf, mlb)
    else:
        print("file not found")


load_data("train_store.pkl", "mlb_data.pkl")

sql = "SELECT * FROM products"
cursor.execute(sql)
data = cursor.fetchall()

print("loading data..")
# data_1
tex = []
for row in data:
    tex.append([row[3]])
    tex.append([unidecode(row[3])])
for i in range(0, len(tex)):
    vectorizer = CountVectorizer()
    vectorizer.fit(tex[i])
    tex[i] = vectorizer.get_feature_names()

new_mlb = MultiLabelBinarizer()
v = new_mlb.fit_transform(tex)

# data2
s1 = []
for (row) in data:
    s1.append(row[12])
    s1.append(row[12])

print("training...")
new_clf = GaussianNB()
new_clf.fit(v, np.array(s1))
pickle.dump(new_clf, open("train_store.pkl", 'wb'))
pickle.dump(new_mlb, open("mlb_data.pkl", 'wb'))
predict_word(new_clf, new_mlb)
print("training complete")

cnx.close()
