import pickle
from pathlib import Path

import mysql.connector
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

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


def predict_word(clf, mlb):
    while True:
        try:
            s = input("Enter something: ")
            input_ = [s]
            vectorizer_ = CountVectorizer()
            vectorizer_.fit_transform(input_)
            list_input = vectorizer_.get_feature_names()
            print(mlb)
            result_array = []
            check_correct = 0
            count_true = 0
            for i in range(0, len(mlb)):
                for j in range(0, len(list_input)):
                    if mlb[i] == list_input[j]:
                        result_array.append(1)
                        check_correct = 1
                        count_true = count_true + 1
                        print("index: " + str(i + 1))
                        print("true: " + mlb[i])
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


load_data("train_store_2.pkl", "mlb_data_2.pkl")

sql = "SELECT * FROM products where category_id != 9999"
cursor.execute(sql)
data = cursor.fetchall()

print("loading data..")
# data_1z
tex = []

for row in data:
    tex.append(row[3])
    tex.append(unidecode(row[3]))
vectorizer = CountVectorizer()
v = vectorizer.fit_transform(tex)

# data2
s1 = []
for (row) in data:
    s1.append(row[12])
    s1.append(row[12])

print("training...")
# model = make_pipeline(TfidfVectorizer, MultinomialNB())
# model.fit(v.toarray(), np.array(s1))
# labels = model.predict("ban di chuot")
new_clf = MultinomialNB()
new_clf.fit(v.toarray(), np.array(s1))
print("training complete")
# pickle.dump(new_clf, open("train_store_2.pkl", 'wb'))
# pickle.dump(vectorizer.get_feature_names(), open("mlb_data_2.pkl", 'wb'))
predict_word(new_clf, vectorizer.get_feature_names())
cnx.close()
