import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
#  print all matrix
import numpy

numpy.set_printoptions(threshold=numpy.nan)


def init_pipeline():
    pipeline = Pipeline([
        ("tdf", TfidfVectorizer()),
        ("cver", CountVectorizer()),
        ("dtc", tree.DecisionTreeClassifier())
    ])
    return pipeline


def processor():
    df = pd.read_csv("data/products.csv")
    train_X, test_X, train_Y, test_Y = train_test_split(df["name"], df["category"], test_size=0.2, random_state=1)

    # pipeline =  init_pipeline()
     # pipeline.fit(train_X,train_Y)
    #
    tdf = TfidfVectorizer()
    dtc = tree.DecisionTreeClassifier()
    model = make_pipeline(tdf,
                          dtc)
    model.fit(train_X, train_Y)
    # print(dict(zip(dtc.classes_, tdf.idf_)))
    pre = model.predict(test_X)
    print(accuracy_score(test_Y, pre))
    # print(confusion_matrix(test_Y, pre))
    print("training complete")

    while True:
        print("Enter something: ")
        tmp = input()
        pred = model.predict([tmp])
        pred_ = model.predict_proba([tmp])
        print(pred_)
        print(pred)


if __name__ == '__main__':
    processor()
