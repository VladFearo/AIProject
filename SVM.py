import random

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

digits = datasets.load_digits()
print(len(digits.data))
clf = svm.SVC(gamma=0.001, C=100)
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)
number = random.randint(0,1797)
test_data = digits.data[number].reshape(1, -1)
prediction = clf.predict(test_data)
print('Prediction:', prediction)
plt.imshow(digits.images[number], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y, prediction)}\n"
)