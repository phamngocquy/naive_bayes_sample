from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import mysql.connector
from unidecode import unidecode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

config = {
    'user': 'root',
    'password': 'quyquy97',
    'host': 'localhost',
    'database': 'price_management',
    'raise_on_warnings': True,
}
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(buffered=True)

sql = "SELECT * FROM products where is_active = 1 AND category_id != 9999"
cursor.execute(sql)
data = cursor.fetchall()

print("loading data..")
# data_1
tex = []
for row in data:
    tex.append(row[3])
    tex.append(unidecode(row[3]))

# data2
s1 = []
for (row) in data:
    s1.append(row[12])
    s1.append(row[12])

train_X, test_X, train_Y, test_Y = train_test_split(tex, s1, test_size=0.2, random_state=1)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_X, train_Y)
pre = model.predict(test_X)
print(accuracy_score(test_Y, pre))
print("training complete")

while True:
    print("Enter something: ")
    tmp = input()
    pred = model.predict([tmp])
    cursor.execute("SELECT name FROM categories WHERE  id = " + str(pred[0]))
    final_data = cursor.fetchall()
    print(final_data)
    print(pred)

# data = fetch_20newsgroups()
# categories = ['talk.religion.misc', 'soc.religion.christian',
#               'sci.space', 'comp.graphics']
#
# train = fetch_20newsgroups(subset='train', categories=categories)
# test = fetch_20newsgroups(subset='test', categories=categories)
# # print(train.data[5])
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#
# model.fit(train.data, train.target)
# # labels = model.predict(test.data)
#
#
#
# pred = model.predict(["sending a payload to the ISS"])
# print(train.target_names[pred[0]])
