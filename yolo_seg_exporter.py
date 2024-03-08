import cv2
import os
import time
import json
import numpy as np
import random

videosPath = "/home/nathan/Desktop/ear_stalk_detection/stalk_videos/stalk_rgb_training_videos"
videosPath = "/home/nathan/Desktop/ear_stalk/stalk_rgb_training_videos/stalk_rgb_training_videos"
videosPaths = ["/media/nathans/AE8D-87D2/indot/converted_videos", 
				"/media/nathans/AE8D-87D2/indot/beta/5-25-thru-5-25", "/media/nathans/AE8D-87D2/indot/beta/6-16-thru-6-21"]
videosPaths =  ["/media/nathans/AE8D-87D2/bee/beeVideos" ]

# videosPath = "/media/nathan/AE8D-87D2/indot/converted_videos/converted_again"
# videosPath = "/media/nathan/AE8D-87D2/indot/epsilon"
# videosPath = "/media/nathan/AE8D-87D2/indot/eta"
# videosPath = "/media/nathan/AE8D-87D2/indot/zeta"
# videosPath = "/media/nathan/AE8D-87D2/indot/beta/5-25-thru-5-25"
# videosPath = "/media/nathan/AE8D-87D2/indot/beta/6-16-thru-6-21"
labelNames = ["yolo_seg"]
# labelNames = ["yolo_seg_obj"]
# labelNames = ["yolo_seg_roadseg"]
# labelNames = ["yolo_seg_shoulderLane"]
# labelNames = ["yolo_seg_frame"]

classesUsed = [0,1,2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
# classesUsed = [0, 1, 2]
classesUsed = [0, 1, 3]#,1,2]
classesExcluded = [3]
classNames = ["stalk"]
classesFound = set()
labelNamesDict = {-1: "erase", 0: "Road", 1: "grass", 2: "mower", 3: "sign",  6: "vehicle", 
                        7: "person", 16: "cone", 9:"guardrail", 10: "shoulder", 11: "drain", 
                        12: "bridgeBarrels", 13: "post", 14: "trash", 15: "roadkill", 4:"building", 5: "tbd", 8: "gravel", 17: "mailbox"}

# labelNamesDict = {-1: "erase", 0: "shoulderLine", 1: "otherLine", 2: "roadEdge"}

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

valDataSplit = 0.15

imageNum = 1

imageSize = 640

actuallySave = True 
pad = True

savePath = os.path.join(videosPaths[0], labelNames[0] + "_yolo_" + str(int(time.time())))


if actuallySave:
	print("saving at", savePath)
	os.mkdir(savePath)
	os.mkdir(os.path.join(savePath, "dataset"))

	os.mkdir(os.path.join(savePath, "dataset", "images"))
	os.mkdir(os.path.join(savePath, "dataset", "labels"))


def compute_intersection(p1, p2, edge):
    # This function computes the intersection point between a polygon edge and a clipping edge
    # p1, p2: Points of the polygon edge
    # edge: The clipping edge
    # Returns the intersection point as (x, y)

    # Clipping rectangle boundaries
    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    
    # Calculate differences
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dx == 0 and dy == 0:
        return p1 # The edge is a point
    
    # Calculate intersection with each boundary
    if edge == 'left':
        x = x_min
        y = p1[1] + dy * (x - p1[0]) / dx
    elif edge == 'right':
        x = x_max
        y = p1[1] + dy * (x - p1[0]) / dx
    elif edge == 'bottom':
        y = y_min
        x = p1[0] + dx * (y - p1[1]) / dy
    elif edge == 'top':
        y = y_max
        x = p1[0] + dx * (y - p1[1]) / dy
    
    return (x, y)

def inside(point, edge):
    # Check if a point is inside the clipping boundary for a given edge
    if edge == 'left':
        return point[0] >= 0
    elif edge == 'right':
        return point[0] <= 1
    elif edge == 'bottom':
        return point[1] >= 0
    elif edge == 'top':
        return point[1] <= 1

def sutherland_hodgman_polygon_clipping(polygon):
    # Clip a polygon against all four edges of the clipping rectangle
    edges = ['left', 'right', 'bottom', 'top']
    
    clipped_polygon = polygon
    
    for edge in edges:
        input_list = clipped_polygon
        clipped_polygon = []
        
        for i in range(len(input_list)):
            current_point = input_list[i]
            prev_point = input_list[i-1]
            
            if inside(current_point, edge):
                if not inside(prev_point, edge):
                    intersection = compute_intersection(prev_point, current_point, edge)
                    clipped_polygon.append(intersection)
                clipped_polygon.append(current_point)
            elif inside(prev_point, edge):
                intersection = compute_intersection(prev_point, current_point, edge)
                clipped_polygon.append(intersection)
    
    return clipped_polygon

for videosPath in videosPaths:
	allFiles = os.listdir(videosPath)



	for file in allFiles:
		if file[-4::].lower() in [".avi", ".mov", ".MOV", ".mp4", ".mkv", ".png", ".jpg", "jpeg"]:
			
			realScale = 1
			for labelName in labelNames:
				jsonName =  file[0:-4] + labelName + ".json"
				if jsonName in allFiles:
					print("found json", jsonName)
					with open(videosPath + "/" + jsonName) as json_file:
						allPolys = json.load(json_file)
						if "classInfo" in allPolys:
						   del allPolys["classInfo"]

					if "remade" in jsonName:
						realScale = 2.5

					if file[-4::] in [".png", ".jpg", "jpeg"]:
						pass
					else:
						cap = cv2.VideoCapture(videosPath + "/" + file)
					for frame in allPolys:
						if file[-4::] in [".png", ".jpg", "jpeg"]:
							img = cv2.imread(os.path.join(videosPath, file))
						else:
							cap.set(1, int(frame))
							_, img = cap.read()


						h, w = img.shape[0:2]

						imgs = []
						regions = []

						if not pad:
							imgs.append(img)
							regions.append([0,0,w,h])
						elif h > w * 1.5 and len(allPolys[frame]) > 5:
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

						elif w > h * 1.5 and len(allPolys[frame]) > 5:
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
							im[:, space:space+w] = img
							imgs = [im]
							regions.append([-space, 0, h+space, h])
							# print('h>w', space)

						elif w > h:
							im = np.zeros((w, w, 3), np.uint8)
							space = random.randint(0, w-h)
							im[space:space+h, :] = img
							imgs = [im]
							regions.append([0, -space, w, w+space])

						else:
							imgs = [img]
							regions = [[0,0, w, h]]
							continue
						print(len(imgs), "images")

						for img, region in zip(imgs, regions):
							# print(region)
							# h_new, w_new = region[3]-region[1], region[2]-region[0]
							h_new, w_new = img.shape[1], img.shape[0]
							img = cv2.resize(img, (imageSize, imageSize), interpolation = cv2.INTER_AREA)
							if not actuallySave:
								imgBlack = np.zeros((imageSize, imageSize, 3), np.uint8)
							
							classesShown = set()
							labelText = ""
							for poly in allPolys[frame]:
								ptList = []
								ptListNorm = []

								if type(poly[-1]) == int:
									lastVal = -1
								elif type(poly[-2]) == int:
									lastVal = -2

								c = poly[lastVal]
								if c not in classesUsed:
									continue
								classesShown.add(c)
								# c = 0

								# if not actuallySave:
								#   print(labelNamesDict[c])
								
								classesFound.add(c)

							
								

								good = False
								for pt in poly[0:lastVal]: # make sure it is in the image of interest
									if 1 >= (pt[0]-region[0])/realScale/w_new >= 0 and 1 >= (pt[1]-region[1])/realScale/h_new >= 0:
										good = True
										break
								if not good:
									# print("no good")
									# c = 3
									continue

								labelText += str(c) + " "


								lastPt = [-1, -1]

								p = [poly[lastVal-1]] + poly[0:lastVal]
								pp = []
								for pt in p:
									x = (pt[0]-region[0])/realScale/w_new
									y = (pt[1]-region[1])/realScale/h_new
									pp.append([x,y])
								p = sutherland_hodgman_polygon_clipping(pp)
								# p = clip_polygon(p, 0, 0)
								for pt in p:
									
									x = pt[0]
									y = pt[1]

									# x = (pt[0]-region[0])/realScale/w_new
									# y = (pt[1]-region[1])/realScale/h_new

									
										

									# print(x, y)


									x = max(min(x, 1), 0)
									y = max(min(y ,1), 0)
									if lastPt[0] != x or lastPt[1] != y:
										# does not account for when it goes out of the image and the line comes back
										labelText += " " + str(x) + " " + str(y)

										ptList += [[int(x*img.shape[1]), int(y*img.shape[0])]]
										if not actuallySave:
											cv2.rectangle(img, (int(x*img.shape[1]), int(y*img.shape[0])), (int(x*img.shape[1]), int(y*img.shape[0])), (0,0,0), 10)
							
									lastPt = [x, y]

									# ptList += [[int(pt[0]/w*imageSize/realScale), int(pt[1]/h*imageSize/realScale)]]

									# # does not account for when it goes out of the image and the line comes back
									# labelText += " " + str(max(min(pt[0]/realScale/w, 1), 0)) + " " + str(max(min(pt[1]/realScale/h,1),0))

								labelText += "\n"

								if not actuallySave:
									colors = [
									(255, 0, 0),     # Bright Blue
									(0, 255, 0),     # Bright Green 
									(255, 0, 255),   # Magenta
									(0, 0, 255),     # Bright Red
									(255, 255, 0),   # Cyan
									(0, 255, 255),   # Yellow
									(0, 165, 255),   # Orange
									(128, 0, 128),   # Purple
									(203, 192, 255), # Pink
									(128, 128, 0),   # Teal
									(0, 0, 128),     # Maroon
									(0, 128, 0),     # Dark Green
									(128, 0, 0),     # Navy Blue
									(0, 128, 128),   # Olive
									(128, 128, 128)  # Grey
									]
									cv2.polylines(img, [np.array(ptList)], True, colors[c%len(colors)], 3)
									cv2.fillPoly(imgBlack, [np.array(ptList)], colors[c%len(colors)])

							good = True
							for c in classesShown:
								if c in classesExcluded:
									good = False
									break
							if not good:
								print("Excluding image")
								continue
							# print(labelText)
							if len(labelText) > 0:
								labelText=labelText[0:-1]

								imgSaveName = "img" + str(imageNum) + ".jpg"

								if actuallySave:
									cv2.imwrite(os.path.join(savePath, "dataset", "images", imgSaveName), img)

									with open(os.path.join(savePath, "dataset", "labels", "img" + str(imageNum) + ".txt"), 'w') as fileHandle:
										fileHandle.write(labelText)
										fileHandle.close()
								else:
									imgBlack = cv2.addWeighted(imgBlack, 0.3, img, 1, 0, img)

								if imageNum % 5 == 0 or not actuallySave:
									cv2.imshow("img", img)
									if actuallySave:
										k=cv2.waitKey(1)
									else:
										# print(labelText)
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
	for i in range(imageNum-1):
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
	coco128yaml += "nc: " + str(len(classesFound)) + "\nnames: " + str(classNames) + " \n"


	coco128yaml += """yolo:


"""




	with open(os.path.join(savePath, "trainInfo.yaml"), 'w') as fileHandle:
		fileHandle.write(coco128yaml)
		fileHandle.close()