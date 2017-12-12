#linear classificarion
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


def extract_color_histogram (image, bin=(8,8,8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BFR2HV)
	hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])

	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist,hist)

	#return the flattened his as the future vector
	return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
 
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the data matrix and labels list
data = []
labels = []

#loop over input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
 
	# extract a color histogram from the image, then update the
	# data matrix and labels list
	hist = extract_color_histogram(image)
	data.append(hist)
	labels.append(label)
 
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

le = LabelEncoder()
labels = le.fit_transform(labels)
#data split
print ("[INFO] constructing training/testing split...")
(trainData,testData,trainLabels,testLabels) = train_test_split(np.array(data), labels, test_size=0.25,random_state=42)

#train linear classifier
print("[INFO] training linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

#evaluate classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
print(classification_report(testLabels,predictions,target_names=le.classes_))






