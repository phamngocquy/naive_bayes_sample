import numpy as np
from unidecode import  unidecode
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(X)
print(Y)
print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[100, 1]]))

your_text = "Nguyễn Trọng Đăng Trình"
your_non_accent_text = unidecode(your_text)
print(your_non_accent_text)
