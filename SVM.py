import time

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

start_time = time.perf_counter()

digits = datasets.load_digits()
print(len(digits.data))
clf = svm.SVC(gamma=0.001, C=100)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

clf.fit(X_train, y_train)
end_time = time.perf_counter()

run_time = end_time - start_time
print(f"Runtime: {run_time:.4f} seconds")
accuracy = clf.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(accuracy))
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


plt.show()
