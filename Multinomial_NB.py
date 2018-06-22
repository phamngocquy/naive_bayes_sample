import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB


class processor():
    df = pd.read_csv("/home/haku/PycharmProjects/DemoOnce/data/products.csv")
    train_x, test_x, train_y, test_y = train_test_split(df["name"], df["category"], test_size=0.2, random_state=1)
    classifier = LabelPowerset(MultinomialNB())
    classifier.fit(train_x, train_y)
    pre = classifier.predict(test_x)
    print(accuracy_score(test_y, pre))


if __name__ == '__main__':
    processor()
