import cv2
import os
import numpy as np

# readDir = "/home/nathan/Desktop/ear_stalk_detection/ear_videos/ear_rgb_training_videos/yolo_earSlant/train"
readDir = "/home/nathan/Desktop/ear_stalk_detection/yolo_oob/train"
imgDir = readDir+"/images/"
textDir = readDir+"/labelTxt/"

fileNames = os.listdir(imgDir)

for fName in fileNames:

	with open(textDir + fName[0:-4] + '.txt') as f:
		lines = f.readlines()
		print(lines)
		allPolys = []
		for line in lines:
			val = ""
			polyPts = []
			for c in line:
				if c == " ":
					try:
						polyPts += [int(float(val))]
					except:
						pass
					val = ""
				else:
					val += c
			allPolys += [np.array([[polyPts[0], polyPts[1]], [polyPts[2], polyPts[3]], [polyPts[4], polyPts[5]], [polyPts[6], polyPts[7]]], np.int32)]

	img = cv2.imread(imgDir + fName)
	
	for pts in allPolys:
		# print(pts)
		img = cv2.polylines(img, [pts], True, (255,255,0), 3)

	cv2.imshow("f", img)
	k=cv2.waitKey(0)

	if k == 27:
		break