import pandas as pd
from sklearn.naive_bayes import MultinomialNB


class Muitinomial_Classifitaion(object):
    def __init__(self):
        self.test = None

    @staticmethod
    def get_data():
        df = pd.read_csv("/home/haku/PycharmProjects/DemoOnce/data/products.csv")
        clf = MultinomialNB()
        clf.fit(df["name"], df["category"])


if __name__ == '__main__':
    mNB = Muitinomial_Classifitaion()
    mNB.get_data()
