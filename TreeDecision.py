import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def processor():
    df = pd.read_csv("data/products.csv")
    train_X, test_X, train_Y, test_Y = train_test_split(df["name"], df["category"], test_size=0.2, random_state=1)
    model = make_pipeline(TfidfVectorizer(),
                          tree.DecisionTreeClassifier(min_samples_split=2, min_weight_fraction_leaf=0))
    model.fit(train_X, train_Y)
    pre = model.predict(test_X)
    print(accuracy_score(test_Y, pre))
    print("training complete")

    while True:
        print("Enter something: ")
        tmp = input()
        pred = model.predict([tmp])
        print(pred)


if __name__ == '__main__':
    processor()
