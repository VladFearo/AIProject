import time

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split

# Start time counter
start_time = time.perf_counter()
# Loading digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
# Creating SVM object
clf = svm.SVC(gamma=0.001, C=1, kernel='rbf')
#Spliting the set into 50:50 training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
# Teaching the model on the training set
clf.fit(X_train, y_train)
# Stopping Clock
end_time = time.perf_counter()

run_time = end_time - start_time
print(f"Runtime: {run_time*1000:.2f} ms")
# Test the model on the test data
accuracy = clf.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(accuracy))
predicted = clf.predict(X_test)

# Showing 4 pictures of numbers with their prediction
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
 # Creating and printing out a report of the model success and accuracy on the testing samples
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
# Creating and showing the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")


plt.show()
