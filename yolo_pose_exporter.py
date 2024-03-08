import cv2
import os
import time
import json
import numpy as np
import random



videosPaths = ["/home/nathan/Desktop/ear_stalk_detection/datasets/ear/rgb_training_videos_big_pc"]
labelName = "ear"
# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_rgb_training_videos_numbered"
# labelName = "stalk"

# videosPath="/home/nathan/Desktop/ear_stalk_detection/datasets/ear/rgb_testing_videos"
# labelName = "earTest2"

# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/unreal/202310021214-RGB"
# labelName = "unreal"


# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_testing_videos/lateharvest/f75/gopro_f75"
# labelName = "combine_stalk"

# # numKeypoints = 3
# videosPaths = ["/home/nathan/Desktop/indot/beta/5-25-thru-5-25", "/home/nathan/Desktop/indot/beta/6-16-thru-6-21"]
# labelName = "road"

# videosPaths = ["/media/nathan/AE8D-87D2/bee/beeVideos"]
# labelName = "bee"
"""
# training_foler/
# | -- dataset/
# |    | -- images/
# |    |    | -- img1.jpg
# |    | -- labels/
# |    |    | -- img1.txt
# |    | -- train.txt
# |    | -- val.txt
# | -- file.yaml


labels (img1.txt):
<object-class> <x1> <y1> <x2> <y2> ...


train.txt
dataset/images/img1.jpg
dataset/images/img2.jpg
dataset/images/img3.jpg

"""

numKeypoints = 2


valDataSplit = 0.15

imageNum = 1

imageSize = 640

actuallySave = True

useVis = False#True

savePath = os.path.join(videosPaths[0], labelName + "_yolo_keypoint_" + str(int(time.time())))

random.seed(0)

if actuallySave:
	print("saving at", savePath)
	os.mkdir(savePath)
	os.mkdir(os.path.join(savePath, "dataset"))

	os.mkdir(os.path.join(savePath, "dataset", "images"))
	os.mkdir(os.path.join(savePath, "dataset", "labels"))
else:
	print("not saving")






for videosPath in videosPaths:
	allFiles = os.listdir(videosPath)



	for file in allFiles:
		if file[-4::].lower() in [".avi", ".mov", ".mp4", ".mkv", ".jpg", ".png"]:

			# if "july" not in file and "Aug" not in file:
			# 	continue
			jsonName =  file[0:-4] + labelName + ".json"
			if jsonName in allFiles:
				print("found json", jsonName)
				with open(videosPath + "/" + jsonName) as json_file:
					allRects = json.load(json_file)

				if file[-4::].lower() not in [".jpg", ".png"]:
					cap = cv2.VideoCapture(videosPath + "/" + file)
			
					
				for frame in allRects:

					if file[-4::].lower() not in [".jpg", ".png"]:
						cap.set(1, int(frame)-1)
						ret, img = cap.read()
						if not ret:
							 break
					else:
						img = cv2.imread(videosPath + "/" + file)



					

					h, w = img.shape[0:2]

					imgs = []
					regions = []
					if h > w * 1.5 and len(allRects[frame]) > 5:
						numFit = int(h/w) + 1
						maxAmt = w*numFit
						remainder = maxAmt - h
						overlap = remainder / (numFit-1)
						step = int(w - overlap)
						y = 0
						while y < w:
							imgs.append(img[y:y+w, :].copy())
							regions.append([0,y, w, y+w])
							y+=step
						# continue

					elif w > h * 1.5 and len(allRects[frame]) > 5:
						# print("w/h")
						numFit = int(w/h) + 1
						maxAmt = h*numFit
						remainder = maxAmt - w
						overlap = remainder / (numFit-1)
						step = int(h - overlap)
						x = 0
						while x < h:
							imgs.append(img[:, x:x+h].copy())
							regions.append([x,0, x+h, h])
							x+=step
						# continue

					

					elif h > w:
						im = np.zeros((h, h, 3), np.uint8)
						space = random.randint(0, h-w)
						space = int((h-w)/2)
						im[:, space:space+w] = img
						imgs = [im]
						regions.append([-space, 0, h+space, h])
						# print('h>w', space)

					elif w > h:
						im = np.zeros((w, w, 3), np.uint8)
						space = random.randint(0, w-h)
						space = int((w-h)/2)
						im[space:space+h, :] = img
						imgs = [im]
						regions.append([0, -space, w, w+space])

					else:
						imgs = [img]
						regions = [[0,0, w, h]]
						continue

					for img, region in zip(imgs, regions):
						# print(region)
						# h_new, w_new = region[3]-region[1], region[2]-region[0]
						h_new, w_new = img.shape[0:2]
						img = cv2.resize(img, (imageSize, imageSize), interpolation = cv2.INTER_AREA)

						labelText = ""
						for rectAll in allRects[frame]:
							
							

							classNum = int(rectAll[4]/2)
							if classNum == 1:
								continue
							

							# if int(rectAll[4]/2) == 0:
							# 	labelText += "0"
							# else:
							# 	# continue
							# 	labelText += "1"

							



							x1, y1, x2, y2 = rectAll[0:4]
							x1, y1 = max(min(x1 - region[0], w_new), 0), max(min(y1 - region[1], h_new), 0)
							x2, y2 = max(min(x2 - region[0], w_new), 0), max(min(y2 - region[1], h_new), 0)
							x_center = ((x1+x2)/2) / w_new
							y_center = ((y1+y2)/2) / h_new
							r_w = abs(x2-x1)/w_new
							r_h = abs(y2-y1)/h_new
							rect = [x_center, y_center, r_w, r_h]
							# rect = [(rectAll[0]+rectAll[2])/2/w, (rectAll[1]+rectAll[3])/2/h, abs(rectAll[2]-rectAll[0])/w, abs(rectAll[3]-rectAll[1])/h]
							removeS = 10/imageSize
							if rect[2] <= removeS or rect[3] <= removeS:
								continue
							
							if x_center+r_w <= 0 or x_center-r_w>=1 or y_center+r_h <= 0 or y_center-r_h>=1:
								continue


							pointsToSave = [classNum, rect[0],rect[1],rect[2],rect[3]]

							if numKeypoints > 1:

								binary_str = bin(int(rectAll[5]/2))[2:].zfill(numKeypoints)

							
								kpList = [not bool(int(digit)) for digit in binary_str]

								kps = [0]* numKeypoints
								if len(rectAll) >= 6 + 2*numKeypoints:
									# pass
									for j, i in enumerate(range(numKeypoints)):
										a = ((rectAll[i*2+6]-region[0])/w_new, (rectAll[i*2+7]-region[1])/h_new)
										if 1 < a[0] or a[0] < 0 or 1 < a[1] or a[1] < 0:
											kpList[j] = False
										kps[i] = (max(min(a[0],1),0), max(min(a[1],1),0))

									
								else:
									labelText = ""
									print("unlabeled- skipping")
									break


								vises = [0] * numKeypoints
								for i in range(numKeypoints):
									if kpList[i]:
										vises[i] = 1
								

								if useVis:
									for i in range(numKeypoints):
										pointsToSave += [kps[i][0], kps[i][1], vises[i]]

								else:
									for i in range(numKeypoints):
										pointsToSave += [kps[i][0], kps[i][1]]


							for pt in pointsToSave:
								if pt < 0 or pt > 1:
									print("uh oh")
									print(rectAll)
									print(pointsToSave)
									# exit()
								labelText += " " + str(pt)
					



							labelText += "\n"



							if not actuallySave:
								colors = [(255,0,0), (0,0,255), (0,255,0)]
								cv2.rectangle(img, (int((rect[0]-rect[2]/2)*imageSize), int((rect[1]-rect[3]/2)*imageSize)),
												   (int((rect[0]+rect[2]/2)*imageSize), int((rect[1]+rect[3]/2)*imageSize)), colors[classNum%len(colors)], 1)

								# cv2.arrowedLine(img, (int(kps[i][0]*imageSize), int(keyPoint1[1]*imageSize)),
								# 				   (int(keyPoint2[0]*imageSize+2), int(keyPoint2[1]*imageSize)), (0,255,255), 1)

								if numKeypoints > 1:
									i=0
									lastPt = [-1, -1]
									while i < numKeypoints:

										pt = pointsToSave[i]

										if lastPt[0] != -1:
											if i == numKeypoints-1:
												cv2.arrowedLine(img, (int(lastPt[0]*imageSize), int(lastPt[1]*imageSize)),
													   (int(kps[i][0]*imageSize), int(kps[i][1]*imageSize)), (0,255,255), 1)
											else:
												cv2.line(img, (int(lastPt[0]*imageSize), int(lastPt[1]*imageSize)),
													   (int(kps[i][0]*imageSize), int(kps[i][1]*imageSize)), (0,255,255), 1)

										if vises[i] == 0 and useVis:
											cv2.rectangle(img, (int(kps[i][0]*imageSize), int(kps[i][1]*imageSize)), (int(kps[i][0]*imageSize), int(kps[i][1]*imageSize)), (0,0,255), 5)
										else:
											cv2.rectangle(img, (int(kps[i][0]*imageSize), int(kps[i][1]*imageSize)), (int(kps[i][0]*imageSize), int(kps[i][1]*imageSize)), (255,0,0), 5)
										lastPt = kps[i]
										i+=1



						if len(labelText) > 0:
							# print(labelText)
							labelText=labelText[0:-1]

							imgSaveName = "img" + str(imageNum) + ".jpg"

							if actuallySave:
								cv2.imwrite(os.path.join(savePath, "dataset", "images", imgSaveName), img)

								with open(os.path.join(savePath, "dataset", "labels", "img" + str(imageNum) + ".txt"), 'w') as fileHandle:
									fileHandle.write(labelText)
									fileHandle.close()

							if imageNum % 50 == 0 or not actuallySave:
								cv2.imshow("img", img)
								if actuallySave:
									k=cv2.waitKey(1)
								else:
									k=cv2.waitKey(0)
								if k == 27:
									exit()
								print(imageNum)
							imageNum += 1
							# if imageNum > 200:
							# 	break


if actuallySave:
	trainTextName = ""
	testTextName = ""
	for i in range(imageNum):
		if random.random() > valDataSplit:
			trainTextName += os.path.join(savePath, "dataset", "images", "img" + str(i+1) + ".jpg") + "\n"
		else:
			testTextName += os.path.join(savePath, "dataset", "images", "img" + str(i+1) + ".jpg") + "\n"


	with open(os.path.join(savePath, "dataset", "train.txt"), 'w') as fileHandle:
		fileHandle.write(trainTextName)
		fileHandle.close()


	with open(os.path.join(savePath, "dataset", "val.txt"), 'w') as fileHandle:
		fileHandle.write(testTextName)
		fileHandle.close()


	coco128yaml = "train: " + os.path.join(savePath, "dataset", "train.txt") + "\n"
	coco128yaml += "val: " +  os.path.join(savePath, "dataset", "val.txt") + "\n"
	coco128yaml += "nc: 1\nnames: ['ground']\n"

	coco128yaml += """
# Keypoints
kpt_shape: [2, 3]
# flip_idx: [2, 1, 0] # if keypoint is symmetrical like left/right eye




"""




	with open(os.path.join(savePath, "trainInfo.yaml"), 'w') as fileHandle:
		fileHandle.write(coco128yaml)
		fileHandle.close()