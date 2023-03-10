import time

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# Start time counter
start_time = time.perf_counter()

# Load the digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
# Creating MLP object
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                    solver='lbfgs',  tol=1e-4, random_state=1,
                    learning_rate_init=1e-3, verbose=True)
# Train the model on the training data
mlp.fit(X_train, y_train)
end_time = time.perf_counter()

run_time = end_time - start_time
print(f"Runtime: {run_time*1000:.2f} ms")

# Test the model on the test data
accuracy = mlp.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(accuracy))
predicted = mlp.predict(X_test)
# Showing 4 pictures of numbers with their prediction
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
 # Creating and printing out a report of the model success and accuracy on the testing samples
print(
    f"Classification report for classifier {mlp}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
# Creating and showing the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
