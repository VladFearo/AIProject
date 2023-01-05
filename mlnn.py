import random
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import train_test_split


# Load the digits dataset
digits = load_digits()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Create the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=1e-3)
# Train the model on the training data
mlp.fit(X_train, y_train)


# Test the model on the test data
accuracy = mlp.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(accuracy))
predicted = mlp.predict(X_test)

# for _ in range(10):
#     number = random.randint(898,1797)
#     test_data = digits.data[number].reshape(1, -1)
#     prediction = mlp.predict(X_test)
#     print('Prediction:', prediction)
#     plt.imshow(digits.images[number], cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.show()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {mlp}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


plt.show()