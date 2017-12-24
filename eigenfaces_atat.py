import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.linalg import eigh as E

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def show_image(image):
	image = np.reshape(image, (112, 92))
	plt.imshow(image, cmap='gray')
	plt.show()

train_paths = []
test_paths = []
unknown_paths = []
base_path = '/Users/aadil/Downloads/att_faces/s'

index = -1
index2 = -1
Y_train = np.zeros(190)
Y_test = np.zeros(190)
images_per_person = 5

for t in range(1, 39):
	path = base_path + str(t)
	count = 0
	for filename in os.listdir(path):
		if(count < images_per_person):
			count += 1
			index2 += 1
			Y_train[index2] = t
			train_paths.append(os.path.join(os.path.sep, path, filename))
		else: 
			index += 1
			Y_test[index] = t
			test_paths.append(os.path.join(os.path.sep, path, filename))

X = np.zeros((0, 10304))

for i in range(len(train_paths)):
	image = cv2.imread(train_paths[i])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.array(image)
	image = image.flatten()

	X = np.vstack((X, image))

X = X.T
## Columns of X give the vector for each face 

## avg is the average face vector
avg = np.average(X, axis=1)

## Subtracting average face vector from each face in X
X = (X.T - avg).T

## Getting the eigenvec in ascending order of eigenvals
R = np.matmul(X.T, X)
(eigenvals, eigenvec) = E(R)
eigenvec = np.matmul(X, eigenvec)

## Reverse sort the eigenvec
eigenvec = eigenvec[:, ::-1]
num_eigenvec = 30

## Keeping only the first 30 eigenvec as they account for max variance
eigenvec = eigenvec[:, 0:num_eigenvec]

## Projecting the images to the eigenspace
X_transformed = np.matmul(eigenvec.T, X)

X_test = np.zeros((0, 10304))

for i in range(len(test_paths)):
	image = cv2.imread(test_paths[i])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.array(image)
	image = image.flatten()

	X_test = np.vstack((X_test, image))

X_test = X_test.T
X_test = (X_test.T - avg).T
X_test_transformed = np.matmul(eigenvec.T, X_test)


## Calculating threshold
rast = np.zeros(len(train_paths))
for i in range(len(train_paths)):
	dist = np.zeros(len(train_paths))
	for j in range(len(train_paths)):
		if i != j:
			dist[j] = math.sqrt(np.sum(np.power((np.subtract(X_transformed[:, j], X_transformed[:, i])), 2)))

		else:
			dist[j] = 999999999

	rast[i] = np.amin(dist)

thresh = 0.5 * np.max(rast)

score = 0
for t in range(len(test_paths)):
	dist = np.zeros(len(train_paths))
	for i in range(len(train_paths)):
		l2 = math.sqrt(np.sum(np.power((np.subtract(X_transformed[:, i], X_test_transformed[:, t])), 2)))
		dist[i] = l2


	pred = np.argmin(dist) // images_per_person + 1
	if pred == Y_test[t] and np.argmin(dist) < thresh:
		score += 1

print ("Accuracy = ", score / len(test_paths))


## Testing some models on the data
Y_train = Y_train.T
X_transformed = X_transformed.T
X_test_transformed = X_test_transformed.T
Y_test = Y_test.T

classifiers = [RandomForestClassifier(n_estimators=1000), LogisticRegression(), DecisionTreeClassifier()]

for classifier in classifiers:
	classifier.fit(X_transformed, Y_train)
	print ("Accuracy = ", classifier.score(X_test_transformed, Y_test))


