import cv2
import numpy as np
import os
import json

# unlabeled- skipping. Frame 317 video Aug24_4.mkv
# 22
# unlabeled- skipping. Frame 376 video Aug24_4.mkv


saveName = ["ear", "out of row"]
# saveName = ["earTest2", "out of row"]
# saveName = ["ear3kp", "runt"]

# saveName = ["stalk"]

# saveName = ["unreal", "out of row"]

# saveName = ["ear", "hairs"]

# saveName = ["ground", "sky"]
# saveName = ["combine_stalk"]
# saveName = ["bee", "queen", "rect"]

numKeypoints = 0
keypointVisibility = True
detector=None

if saveName == ["ear3kp", "runt"]:
	#  stalk -> base -> tip
	
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/rgb_training_videos_big_pc"
	numKeypoints = 3
if saveName == ["unreal", "out of row"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/unreal/202310021214-RGB"
	# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/unreal/202309112151_rgb"
	# import run_yolo_keypoint as detector
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/unreal_oct3.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_sep26.pt")
	numKeypoints = 2

if saveName == ["ear", "out of row"]:
	# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/ear_rgb_training_videos_old"
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/rgb_training_videos_big_pc"
	# videosPath = "/home/nathans/Desktop/ear_stalk_detection/datasets/ear/unreal/nov9/rgb_11_9_2023/202311092043"
	# import run_yolo_keypoint as detector
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/train/yolo/runs/pose/train7/weights/best.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/young_ear_keypoint_july21.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_aug9.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_aug18.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/keypoint_jun15.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_keypoint_aug23_2.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_aug24.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_sep26.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_oct6.pt")

	numKeypoints = 2
elif saveName == ["earTest2", "out of row"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/rgb_testing_videos"
	import run_yolo_keypoint as detector
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_aug9.pt")
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_aug18.pt")
	detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/yolo_ear_out_row_keypoint_oct6.pt")
	numKeypoints = 2


elif saveName == ["ear", "hairs"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/super_young_ear_vids"
	# videosPath = "/home/nathan/Desktop/grain_fill/zed_1690208713"

	import run_yolo_keypoint as detector
	detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/yolo_ear/young_ear_keypoint_july21.pt", confidence=0.2)
	numKeypoints = 2
	
elif saveName == ["stalk", "weed"]:
	videosPath = "/home/nathan/Desktop/oscar_ml/dataset"
	numKeypoints = 3

elif saveName == ["ground", "sky"]:
	videosPath = "/home/nathan/Desktop/oscar_ml/dataset"
	numKeypoints = 3

elif saveName == ["stalk"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_rgb_training_videos_numbered"
	# import run_yolo_keypoint as detector
	# detector.setupModel("/home/nathan/Desktop/ear_stalk_detection/models/other/yolo_stalk_angle_aug28.pt", confidence=0.2)

	numKeypoints = 2

elif saveName == ["combine_stalk"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_testing_videos/lateharvest/f75/gopro_f75"

elif saveName == ["bee", "queen", "rect"]:
	numKeypoints = 2
	videosPath = "/media/nathan/AE8D-87D2/bee/beeVideos"
	videosPath = '/home/nathan/Desktop/CV_lab/lab/non_lab/beeVideos'
	videosPath = "/media/nathan/AE8D-87D2/bee/beeVideos"
	import run_yolo_keypoint as detector
	detector.setupModel("/media/nathan/AE8D-87D2/bee/bee_keypoint_feb16_640_pad.pt", confidence=0.2, padding=True)

class Annotator:
	def __init__(self, videosPath, saveName, detector=None, numKeypoints=2, keypointVisibility=True):

		self.windowScale = 1

		self.videosPath = videosPath
		self.saveName = saveName

		self.currentRect = [] # format [[x,y], [x,y], ...]
		self.firstCallback = True
		self.falseKey = -1

		self.keypointVisibility = keypointVisibility

		self.numKeypoints = numKeypoints


		self.openButtons = {"Clear": -2, "Delete": 8, "Next": 108, "Prev": 106, "Next Vid": 47, "Prev Vid": 46, "Next Annot": 93, "Prev Annot": 91, "Last Annot": 92, "Zoom": -6, "Exit": 27}


		self.possibleRects = []


		self.detector = detector
		self.makeDetections = False

		print(detector)
		self.makingArrow = False

		self.justMadeRect = False

		# this script technically supports multiple classes
		self.labelNum = 0
		self.lastLabelNum = 0

		self.mouseDown = False

		self.unsavedChanges = False
		self.zooming = False
		self.zoomRect = []
		self.lastPan = []
		self.startPan = []

		self.selectedInd = -1

		self.crosshair = True

		self.mouseX, self.mouseY = 0, 0

		self.openImgFrame = -2

		# trunc + 2*classNum
		# diff + 2**1*keypoint1hidden + 2**2*keypoint2hidden ...
		self.allRects = {} # format {"Frame #": [[x1, y1, x2, y2, trunc, difficult, keypointX1, keypointY1, keypointX2, keypointY2, keypointXn...], [,,,...]...]

	def manageKeyResponses(self):
		change = 0

		while change == 0:
			k = cv2.waitKey(10)

			# self.openButtons = {"Cancel": 27, "Delete": 8, "Truncated": -3, "Difficult": -4, "class++": -5}


			if self.falseKey != -1: # treat button presses as key presses
				k = self.falseKey
				self.falseKey = -1

			if k == -1: # no response from keyboard
				continue

			if k == -2: # clear all annotations in frame
				if len(self.allRects[str(self.frameNum)]) > 0:
					self.unsavedChanges = True
					self.numVidRects -= len(self.allRects[str(self.frameNum)])
					self.totalRects -= len(self.allRects[str(self.frameNum)])
					self.allRects[str(self.frameNum)] = []
					self.selectedInd = -1
					self.dispImg()

			elif k == -3: # trunc
				classNum = int(self.allRects[str(self.frameNum)][self.selectedInd][4] // 2)
				trunc = int((self.allRects[str(self.frameNum)][self.selectedInd][4]+1) % 2)
				self.allRects[str(self.frameNum)][self.selectedInd][4] = trunc + classNum*2
				self.unsavedChanges = True
				self.dispImg()

			elif k == -4: # difficult
				otherInfo = int(self.allRects[str(self.frameNum)][self.selectedInd][5] // 2)
				difficult = int((self.allRects[str(self.frameNum)][self.selectedInd][5]+1) % 2)
				self.allRects[str(self.frameNum)][self.selectedInd][5] = difficult + otherInfo*2
				self.unsavedChanges = True
				self.dispImg()

			elif k == -5: # class ++
				classNum = int(self.allRects[str(self.frameNum)][self.selectedInd][4] // 2)
				trunc = int((self.allRects[str(self.frameNum)][self.selectedInd][4]) % 2)
				classNum = (classNum + 1) % len(self.saveName)
				self.allRects[str(self.frameNum)][self.selectedInd][4] = trunc + classNum*2
				self.unsavedChanges = True
				self.dispImg()
			elif k == -6: # unzoom/zoom
				if "Zoom" in self.openButtons:
					self.zooming = True
					self.zoomRect = []
					nb = {}
					for i in self.openButtons:
						if i != "Zoom":
							nb[i] = self.openButtons[i]
						else:
							nb["Unzoom"] = self.openButtons[i]
					self.openButtons = nb

				elif "Unzoom" in self.openButtons:
					self.zoomRect = []
					self.zooming = False
					nb = {}
					for i in self.openButtons:
						if i != "Unzoom":
							nb[i] = self.openButtons[i]
						else:
							nb["Zoom"] = self.openButtons[i]
					self.openButtons = nb
				self.dispImg()



			elif k <= -100:

				kpToggle = -(k+100)+1

				ogVal = self.allRects[str(self.frameNum)][self.selectedInd][5]
				diff = ogVal%2
				ogVal = int(ogVal/2)
				binary_str = bin(ogVal)[2:].zfill(self.numKeypoints+1)
				kpList = [bool(int(digit)) for digit in binary_str]

				kpList[kpToggle] = not kpList[kpToggle]

				binary_str = "".join(str(int(b)) for b in kpList)
				self.allRects[str(self.frameNum)][self.selectedInd][5] = int(binary_str, 2)*2 + diff
				

				self.dispImg()
				

			elif k == 27: # escape - either stop making polygon or exit
				if len(self.currentRect) > 0:
					self.currentRect = []
					self.dispImg(drawAmount=1)
				else:
					self.saveJSON()
					if self.vidName[-4::] not in [".jpg", ".png"]:
						self.cap.release() # close current video
					exit()

			elif k == 13: # enter or middle mouse button - finish making 
				self.justMadePoly = True

				if len(self.currentRect) > 3: # need at least 3 points to make polygon

					self.makePoly()

					self.dispImg()

			

				self.currentRect = []

				self.dispImg(drawAmount=0)

			elif k == 47: # / - go to next video
				change = self.maxFrameNum + 10

			elif k == 46: # . - go to previous video
				change = -self.frameNum - 10

			elif k == 93 or k == 9: # ] or tab- next annotation
				frameJump = self.maxFrameNum+10
				for frame in self.allRects:
					if int(frame) > self.frameNum and len(self.allRects[frame]) > 0:
						frameJump = min(int(frame), frameJump)
				if frameJump < self.maxFrameNum+10:
					change = frameJump - self.frameNum
	
			elif k == 91 or k == 9: # [ or caps lock- prev annotation
				frameJump = -1
				for frame in self.allRects:
					if int(frame) < self.frameNum and len(self.allRects[frame]) > 0:
						frameJump = max(int(frame), frameJump)
				if frameJump != -1:
					change = frameJump - self.frameNum

			elif k == 92: # \ - last annotation
				frameJump = -1
				for frame in self.allRects:
					if len(self.allRects[frame]) > 0:
						frameJump = max(int(frame), frameJump)
				if frameJump != -1:
					change = frameJump - self.frameNum

			elif k == 8: # delete
				if len(self.allRects[str(self.frameNum)]) > 0:
					self.allRects[str(self.frameNum)].remove(self.allRects[str(self.frameNum)][self.selectedInd])
					self.selectedInd = -1
					self.numVidRects -= 1
					self.totalRects -= 1
					self.makingArrow = False
					self.unsavedChanges = True

				self.dispImg()

			elif k == 106: # j - go back 5 frames
				change = - 5

			elif k == 108: # l - go forward 5 frames
				change = 5

			elif 58 > k >= 49: # 49=1

				if self.selectedInd >= 0 and self.numKeypoints >= k-49:
					kpToggle = k-49
					ogVal = self.allRects[str(self.frameNum)][self.selectedInd][5]
					diff = ogVal%2
					ogVal = int(ogVal/2)
					binary_str = bin(ogVal)[2:].zfill(self.numKeypoints+1)
					kpList = [bool(int(digit)) for digit in binary_str]

					kpList[kpToggle] = not kpList[kpToggle]

					binary_str = "".join(str(int(b)) for b in kpList)
					self.allRects[str(self.frameNum)][self.selectedInd][5] = int(binary_str, 2)*2 + diff
					self.dispImg()

				elif len(self.saveName) > k-49 and self.selectedInd < 0:
					print("changing class", k-49)
					self.labelNum = k-49
					self.dispImg()

			elif k == 121: # y
				self.makeDetections = not self.makeDetections

			else: # any other key pressed - go forward 1 key

				change = 1

		return change

	def mouseEvent(self, event, x, y, flags, param):
		if event == 1:
			self.mouseDown = True
		elif event == 4:
			self.mouseDown = False

		if event == 0: # moved mouse after making a rectangle. This is to avoid making a second rectangle accidently, which happens sometimes
			self.justMadeRect = False

		if 10 < x/self.windowScale < 200: # cursor in region with buttons
			if event == 4: # mouse up
				for i, buttonName in enumerate(self.openButtons):
					if 5+(i+1)*70 > y/self.windowScale > 10+i*70:
						self.falseKey = self.openButtons[buttonName]
						break
			return

		else:
			minX = 0
			minY = 0
			if not self.zooming and len(self.zoomRect) > 0:
				minX = self.zoomRect[0]
				minY = self.zoomRect[1]


			ogX, ogY = x, y
			
			self.mouseX, self.mouseY = x, y
			# convert x,y mouse position to x, y in image scale
			x = int((x - self.xPad - self.buttonPad) / self.scale) + minX
			y = int((y - self.yPad) / self.scale) + minY

			x = max(min(x, self.openImg.shape[1]), 0)
			y = max(min(y, self.openImg.shape[0]), 0)


					
			if event == 0 and len(self.currentRect) > 0: # mouse moved and making rectangle
				self.currentRect[2] = x
				self.currentRect[3] = y
				self.dispImg(drawAmount=1)
				return

			elif event == 0 and self.makingArrow:
				self.allRects[str(self.frameNum)][self.selectedInd][len(self.allRects[str(self.frameNum)][self.selectedInd])-2] = x
				self.allRects[str(self.frameNum)][self.selectedInd][len(self.allRects[str(self.frameNum)][self.selectedInd])-1] = y
				self.dispImg(drawAmount=1)

			elif event == 0 and len(self.zoomRect) > 0 and self.zooming:
				self.zoomRect[2] = x
				self.zoomRect[3] = y
				self.dispImg(drawAmount=1)




			elif event == 6:
				if abs(ogX-self.startPan[0]) + abs(ogY-self.startPan[1]) < 10 and self.selectedInd != -1: # middle mouse to toggle class
					self.falseKey = -5
				self.lastPan = []
				self.startPan = []


			elif event == 3: # middle mouse button down
				self.lastPan = [ogX, ogY]
				self.startPan = [ogX, ogY]

			elif event == 0 and len(self.lastPan) > 0 and len(self.zoomRect) > 0:
				xChange = self.lastPan[0]- ogX
				yChange = self.lastPan[1]- ogY
				if xChange >= 0:
					if self.zoomRect[2] + xChange > self.openImg.shape[1]:
						xChange = self.openImg.shape[1] - self.zoomRect[2]
				else:
					if self.zoomRect[0] + xChange < 0:
						xChange = 0 - self.zoomRect[0]


				if yChange >= 0:
					if self.zoomRect[3] + yChange > self.openImg.shape[0]:
						yChange = self.openImg.shape[0] - self.zoomRect[3]
				else:
					if self.zoomRect[1] + yChange < 0:
						yChange = 0 - self.zoomRect[1]

				self.zoomRect[0] += xChange
				self.zoomRect[2] += xChange
				self.zoomRect[1] += yChange
				self.zoomRect[3] += yChange


				self.lastPan = [ogX,ogY]

				self.dispImg()
				return



			elif event == 1: # mouse down, make next point of rectangle
				if self.justMadeRect:
					return

				if self.zooming:
					if len(self.zoomRect) == 0:
						self.zoomRect = [x, y, x, y]
					else:
						self.zoomRect = []
					return


				if self.selectedInd != -1:
					rect = self.allRects[str(self.frameNum)][self.selectedInd]
					if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
						if self.makingArrow:

							return
						self.makingArrow = True
						if self.numKeypoints >= 2:
							self.allRects[str(self.frameNum)][self.selectedInd] = self.allRects[str(self.frameNum)][self.selectedInd][0:6] + [x, y, x, y]

						self.dispImg(drawAmount=1)
					elif len(self.currentRect) == 0:
						self.selectedInd = -1
						self.currentRect = [x, y, x, y]


				elif len(self.currentRect) == 0:
					self.currentRect = [x, y, x, y]

				return

			elif event == 4: # mouse up: make rectangle or check clicks

				if self.zooming:
					if len(self.zoomRect) == 4:
						if abs(self.zoomRect[0]-self.zoomRect[2]) > 20 and abs(self.zoomRect[0]-self.zoomRect[2]) > 20:
							self.zoomRect = [min(self.zoomRect[0], self.zoomRect[2]), min(self.zoomRect[1], self.zoomRect[3]), max(self.zoomRect[2], self.zoomRect[0]), max(self.zoomRect[3], self.zoomRect[1])]
							self.zooming = False
							self.dispImg()


					return

				if len(self.currentRect) == 4 and abs(self.currentRect[0]-self.currentRect[2])*abs(self.currentRect[1]-self.currentRect[3]) > 100:
					# add rect
					x1, y1, x2, y2 = self.currentRect
					rectAdd = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), self.labelNum*2, 0]
					self.allRects[str(self.frameNum)] += [rectAdd]
					self.selectedInd = len(self.allRects[str(self.frameNum)]) - 1
					self.numVidRects += 1
					self.totalRects += 1
					self.unsavedChanges = True

				elif self.makingArrow:
					if int((len(self.allRects[str(self.frameNum)][self.selectedInd])-6) / 2) < self.numKeypoints:
						self.allRects[str(self.frameNum)][self.selectedInd] += [x,y]
						# print("new")
					else:
						# print("done")
						self.makingArrow = False
						self.unsavedChanges = True
				else:
					self.selectedInd = -1
					for i, rect in enumerate(self.allRects[str(self.frameNum)]):
						if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
							self.selectedInd = i
					if self.selectedInd == -1:
						for i, rect in enumerate(self.possibleRects):
							if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
								self.selectedInd = i
								self.allRects[str(self.frameNum)] += [rect]
								self.totalRects += 1
								self.numVidRects += 1
								self.selectedInd = len(self.allRects[str(self.frameNum)])-1
								break
						if self.selectedInd != -1:
							self.possibleRects.remove(self.possibleRects[i])

				self.currentRect = []

				self.dispImg()
			
			elif self.crosshair:
				self.dispImg(drawAmount=2)



	def dispImg(self, drawAmount=0):
		self.xPad = int(50*self.windowScale)
		self.yPad = int(50*self.windowScale)
		self.buttonPad = int(200*self.windowScale)

		# get offset of image if you are zoomed in
		minX = 0
		minY = 0
		if len(self.zoomRect) > 0 and not self.zooming:
			minX = self.zoomRect[0]
			minY = self.zoomRect[1]


		if len(self.zoomRect) > 0 and not self.zooming:
			openImg = self.openImg[self.zoomRect[1]:self.zoomRect[3], self.zoomRect[0]:self.zoomRect[2]]
		else:
			openImg = self.openImg.copy()

		if drawAmount < 1: # redraw everything - sometimes this is not necessary, so it can keep the previous image, which is much faster
			self.imgH, self.imgW = self.openImg.shape[0:2]


			# resize the image to take up as much of the window as possible
			self.scale = 1

			h, w = openImg.shape[0:2]
			if h/w > 850/1280:
				self.scale = 850*self.windowScale/h
				openImg = cv2.resize(openImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)
			else:
				self.scale = 1280*self.windowScale/w
				openImg = cv2.resize(openImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)


			# make blank image to show
			drawImg = np.zeros((max(openImg.shape[0]+self.yPad*2, int(950*self.windowScale)), max(openImg.shape[1]+self.xPad*2+self.buttonPad, int(1300*self.windowScale)), 3), np.uint8)


			if self.selectedInd == -1:
				# self.openButtons = {"Clear": -2, "Delete": 8, "Next": 108, "Prev": 106, "Next Vid": 47, "Prev Vid": 46, "Next Annot": 93, "Prev Annot": 91, "Last Annot": 92, "Exit": 27}
				
				if self.zooming or len(self.zoomRect) > 0:
					self.openButtons = {"Clear": -2, "Delete": 8, "Next": 108, "Prev": 106, "Next Vid": 47, "Prev Vid": 46, "Next Annot": 93, "Prev Annot": 91, "Last Annot": 92, "Unzoom": -6, "Exit": 27}

				else:
					self.openButtons = {"Clear": -2, "Delete": 8, "Next": 108, "Prev": 106, "Next Vid": 47, "Prev Vid": 46, "Next Annot": 93, "Prev Annot": 91, "Last Annot": 92, "Zoom": -6, "Exit": 27}

				txtVals = ["Label Type: " + str(self.labelNum), "Rects labeled: " + str(self.numVidRects) + " (" + str(self.totalRects) + ")", "Frames labeled: " + str(self.numVidAnnotations) + " (" + str(self.totalAnnotations) + ")", "Vid: " + self.vidName, "Vid #: " + str(self.vidIndex) + "/" + str(len(self.videos)), "Frame: " + str(self.frameNum) + "/" + str(self.maxFrameNum)]
			else:
				self.openButtons = {"Cancel": 27, "Delete": 8, "Truncated": -3, "Difficult": -4, "class++": -5}
				rect = self.allRects[str(self.frameNum)][self.selectedInd]
				classNum = int(rect[4]//2)
				txtVals = ["Class: " + self.saveName[classNum], "Trunc: " + str(rect[4]%2 == 1), "Diff: " + str(rect[5]%2 == 1)]
				if self.keypointVisibility:
					for i in range(self.numKeypoints):
						self.openButtons["kp " + str(i) + " vis"] = -100 - i



			# draw buttons
			for i, buttonName in enumerate(self.openButtons):
				cv2.rectangle(drawImg, (int(5*self.windowScale), int(self.windowScale*(10+i*70))), (self.buttonPad, int(self.windowScale*(5+(i+1)*70))), (255,255,0), -1)
				cv2.rectangle(drawImg, (int(10*self.windowScale), int(self.windowScale*(15+i*70))), (self.buttonPad-int(self.windowScale*5), int(self.windowScale*((i+1)*70))), (255,255,255), -1)
				cv2.putText(drawImg, buttonName, (15, (i+1)*70-20), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (255,0,0), max(int(2*self.windowScale),1), cv2.LINE_AA)

			# draw info text
			for i, val in enumerate(txtVals):
				cv2.putText(drawImg, val, (15, int(self.windowScale*(840+i*20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5*self.windowScale, (255,255,255),  max(int(1*self.windowScale),1), cv2.LINE_AA)

			

			# add the image to the overall canvas
			drawImg[self.yPad:openImg.shape[0]+self.yPad, self.xPad+self.buttonPad:openImg.shape[1]+self.xPad+self.buttonPad] = openImg
			self.baseImg = drawImg.copy()

		if drawAmount < 2:

			drawImg = self.baseImg.copy()

			# draw the current rectangle
			if len(self.currentRect) > 0:
				cv2.rectangle(drawImg, (int((self.currentRect[0]-minX)*self.scale+self.buttonPad+self.xPad), int((self.currentRect[1]-minY)*self.scale+self.yPad)), 
										(int((self.currentRect[2]-minX)*self.scale+self.buttonPad+self.xPad), int((self.currentRect[3]-minY)*self.scale+self.yPad)), 
										(0,0,0), 1)

			if len(self.zoomRect) > 0 and self.zooming:
				cv2.rectangle(drawImg, (int((self.zoomRect[0]-minX)*self.scale+self.buttonPad+self.xPad), int((self.zoomRect[1]-minY)*self.scale+self.yPad)), 
										(int((self.zoomRect[2]-minX)*self.scale+self.buttonPad+self.xPad), int((self.zoomRect[3]-minY)*self.scale+self.yPad)), 
										(0,0,0), 1)

			labelColors = [(255,0,0), (255,255,0), (0,0,255), (255,0,255)]
			for i, rect in enumerate(self.allRects[str(self.frameNum)]):
				if i == self.selectedInd:
					color = (0,255,0)
				else:
					classNum = int(rect[4]//2)

					color = labelColors[classNum % len(labelColors)]



				cv2.rectangle(drawImg, (int((rect[0]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[1]-minY)*self.scale+self.yPad)), 
									(int((rect[2]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[3]-minY)*self.scale+self.yPad)), 
									color, 2)

				if self.keypointVisibility:
					binary_str = bin(int(rect[5]/2))[2:].zfill(self.numKeypoints+1)
					kpList = [not bool(int(digit)) for digit in binary_str]

				hasKeypoints = False
				if len(rect) > 9 and self.numKeypoints == 2:
					if self.keypointVisibility:

						xxx = (int((rect[6]-minX)*self.scale+self.buttonPad+self.xPad))
						yyy = (int((rect[7]-minY)*self.scale+self.yPad))
						if kpList[1]:
							cv2.rectangle(drawImg, (xxx,yyy), (xxx,yyy), (0,255,0), 10)
						else:
							cv2.rectangle(drawImg, (xxx,yyy), (xxx,yyy), (0,0,255), 10)

						xxx = (int((rect[8]-minX)*self.scale+self.buttonPad+self.xPad))
						yyy = (int((rect[9]-minY)*self.scale+self.yPad))
						if kpList[2]:
							cv2.rectangle(drawImg, (xxx,yyy), (xxx,yyy), (0,255,0), 10)
						else:
							cv2.rectangle(drawImg, (xxx,yyy), (xxx,yyy), (0,0,255), 10)


					cv2.arrowedLine(drawImg, (int((rect[6]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[7]-minY)*self.scale+self.yPad)), 
									(int((rect[8]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[9]-minY)*self.scale+self.yPad)), 
									color, 2)
					hasKeypoints = True

				elif len(rect) > 7:
					kpNum=6
					lastPt = [rect[6]+1, rect[7]+1]
					vi = 0
					while kpNum < len(rect):
						if self.keypointVisibility:

							if kpList[vi+1]:
								cv2.rectangle(drawImg, (int((rect[kpNum]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[kpNum+1]-minY)*self.scale+self.yPad)), 
														(int((rect[kpNum]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[kpNum+1]-minY)*self.scale+self.yPad)), (0,255,0), 10)
							else:
								cv2.rectangle(drawImg, (int((rect[kpNum]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[kpNum+1]-minY)*self.scale+self.yPad)), 
														(int((rect[kpNum]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[kpNum+1]-minY)*self.scale+self.yPad)), (0,0,255), 10)
							vi+=1
						if len(rect) - kpNum == 2:
							cv2.arrowedLine(drawImg, (int((lastPt[0]-minX)*self.scale+self.buttonPad+self.xPad), int((lastPt[1]-minY)*self.scale+self.yPad)), 
									(int((rect[kpNum]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[kpNum+1]-minY)*self.scale+self.yPad)),  
									color, 2)
						cv2.line(drawImg, (int((rect[kpNum]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[kpNum+1]-minY)*self.scale+self.yPad)), 
									(int((lastPt[0]-minX)*self.scale+self.buttonPad+self.xPad), int((lastPt[1]-minY)*self.scale+self.yPad)), 
									color, 2)
						lastPt = [rect[kpNum], rect[kpNum+1]]
						kpNum+=2
					hasKeypoints = True
				if not hasKeypoints and numKeypoints > 0 and color != (0,255,0):
					cv2.rectangle(drawImg, (int((rect[0]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[1]-minY)*self.scale+self.yPad)), 
									(int((rect[2]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[3]-minY)*self.scale+self.yPad)), 
									(0,0,255), 2)



				midX, midY = int(((rect[0]+rect[2])/2-minX)*self.scale)+self.buttonPad+self.xPad, int(((rect[1]+rect[3])/2-minY)*self.scale)+self.yPad
				if rect[4]%2 == 1:
					cv2.putText(drawImg, "T", (midX-10, midY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
				if rect[5]%2 == 1:
					cv2.putText(drawImg, "D", (midX+10, midY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

			for i, rect in enumerate(self.possibleRects):
				color = (0,255,255)

				cv2.rectangle(drawImg, (int((rect[0]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[1]-minY)*self.scale+self.yPad)), 
									(int((rect[2]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[3]-minY)*self.scale+self.yPad)), 
									color, 1)

				if len(rect) > 6:
					cv2.arrowedLine(drawImg, (int((rect[6]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[7]-minY)*self.scale+self.yPad)), 
										(int((rect[8]-minX)*self.scale+self.buttonPad+self.xPad), int((rect[9]-minY)*self.scale+self.yPad)), 
										color, 1)
			if self.crosshair:
				self.baseImg2 = drawImg.copy()
		else:
			drawImg = self.baseImg2.copy()

			if self.mouseX > self.buttonPad and self.crosshair:
				cv2.line(drawImg, (self.mouseX, self.yPad), (self.mouseX, drawImg.shape[0]-self.yPad), (0,0,0),1)
				cv2.line(drawImg, (self.xPad+self.buttonPad, self.mouseY), ( drawImg.shape[1]-self.yPad, self.mouseY), (0,0,0),1)



		cv2.imshow("img", drawImg)


		if self.firstCallback: # allow mouse events
			self.firstCallback = False
			cv2.setMouseCallback("img", self.mouseEvent)



	def runThrough(self):
		allFiles = os.listdir(self.videosPath)

		# for i in allFiles:
		# 	if i[-4::] in [".MOV", ".mov", ".mp4", ".avi", ".mkv"]:
		# 		self.videos += [i]
		# self.videos = sorted(self.videos)
		# get the videos from the files
		self.videos = []
		
		# priorities = ["Aug", "oct", "sep", "Aug", "july", "avi", "jun"]
		priorities = []#["a_queen_IMG_2398", "2024"]
		groups = []
		for i in range(len(priorities)+1):
			groups += [[]]
		print(groups)

		allFiles = sorted(allFiles)
		for i in allFiles:
			if i[-4::].lower() in [".MOV", ".mov", ".mp4", ".avi", ".mkv"]:#, ".jpg", ".png", ".MP4"]:
				added = False
				for n, j in enumerate(priorities):
					if j in i:
						groups[n] += [i]
						added = True
						break
				if not added:
					groups[-1] += [i]


		for g in groups:
			self.videos += g
		self.videos.reverse()

		print(self.videos)

		self.infoJsonName = "info_" + self.saveName[0] + ".json"
		if self.infoJsonName in os.listdir(self.videosPath):
			with open(os.path.join(videosPath, self.infoJsonName)) as json_file:
				self.vidInfo = json.load(json_file)
				self.totalAnnotations = 0
				self.totalRects = 0
				for vidName in self.vidInfo:
					self.totalAnnotations += self.vidInfo[vidName][0]
					self.totalRects += self.vidInfo[vidName][1]
					print(vidName, self.vidInfo[vidName][1])
		else:
			self.totalAnnotations = 0
			self.totalRects = 0
			self.vidInfo = {}



		# go through all the videos
		self.vidIndex = 0
		while self.vidIndex < len(self.videos):

			self.vidName = self.videos[self.vidIndex]

			# load the annotations
			self.allRects = {}
			self.jsonName = self.vidName[0:-4] + self.saveName[0] + ".json"
			if self.jsonName in os.listdir(self.videosPath) and True:
				print("found json", self.jsonName)
				with open(os.path.join(videosPath, self.jsonName)) as json_file:
					self.allRects = json.load(json_file)
					self.numVidAnnotations = len(list(self.allRects.keys()))
					self.numVidRects = 0
					for i in self.allRects:
						self.numVidRects += len(self.allRects[i])
			else:
				self.numVidRects = 0
				self.numVidAnnotations = 0

			# begin video
			if self.vidName[-4::] in [".jpg", ".png"]:
				self.maxFrameNum = 1
			else:
				self.cap = cv2.VideoCapture(os.path.join(self.videosPath, self.vidName))
				self.maxFrameNum = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


			self.frameNum = 0

			for frame in self.allRects:
				# format the rectangles to work
				for i in range(len(self.allRects[frame])):
					rect = self.allRects[frame][i]
					if len(rect) == 4:
						print("bad length, fixing") 
						self.allRects[frame][i] += [0, 0]
						self.unsavedChanges = True


			
			# go through all the frames of the video
			while 0 <= self.frameNum < self.maxFrameNum:
				# if self.openImgFrame != self.frameNum:
				# 	if self.openImgFrame != self.frameNum-1:
				# 		self.cap.set(1, self.frameNum)
				# 	ret, self.openImg = self.cap.read()
				# 	self.openImgFrame = self.frameNum
				
				if self.vidName[-4::] in [".jpg", ".png"]:
					self.openImg = cv2.imread(os.path.join(self.videosPath, self.vidName))
				else:
					self.cap.set(1, self.frameNum-1)
					ret, self.openImg = self.cap.read()
			

				self.makingArrow = False

				
				if self.numVidAnnotations != len(list(self.allRects.keys())):
					self.numVidAnnotations = len(list(self.allRects.keys()))
					self.totalAnnotations = 0
					self.numVidRects = 0
					for vidName in self.vidInfo:
						if vidName != self.vidName:
							self.totalAnnotations += self.vidInfo[vidName][0]
							self.numVidRects += self.vidInfo[vidName][1]
							
					self.totalAnnotations += self.numVidAnnotations

				if str(self.frameNum) not in self.allRects: # get the annotation
					self.allRects[str(self.frameNum)] = []

				self.selectedInd = -1


				


				if self.detector is not None and self.makeDetections:
					possibleRects = self.detector.detectPossible(self.openImg, annotate=False, testSize=640)
					self.possibleRects = []
					for rect in possibleRects:
						found = False
						for rect2 in self.allRects[str(self.frameNum)]:
							if (rect[0]<rect2[2] and rect[2]>rect2[0]) and (rect[1]<rect2[3] and rect[3]>rect2[1]):
								found=True
								break
						if not found:
							# rect = rect[0:6] + [int(rect[6][0]), int(rect[6][1]), int(rect[7][0]), int(rect[7][1])]
							self.possibleRects += [rect]
					# print(self.possibleRects)

				self.dispImg(drawAmount=0) # show the annotation thing
				change = self.manageKeyResponses() # get key responses

				if len(self.allRects[str(self.frameNum)]) == 0: # remove annotations from main dict if there are no annotations
					del(self.allRects[str(self.frameNum)])

				self.frameNum += change
				self.possibleRects = []

			self.saveJSON() # save the data

			# go to the appropriate video
			if self.frameNum < 0:
				self.vidIndex = max(0, self.vidIndex-1)
			elif self.frameNum > 0:
				self.vidIndex = min(len(self.videos)+1, self.vidIndex+1)
			else:
				print("couldn't get max frame or something. Frame number was 0. It might be corrupted.", self.vidName, "max frame", self.maxFrameNum)
				self.vidIndex = min(len(self.videos)+1, self.vidIndex+1)
			self.currentPoly = []
		print("done with all videos")


	def saveJSON(self):
		"""
		Save the annotations
		"""

		polyNames = list(self.allRects.keys())
		for i in polyNames:
			if len(self.allRects[i]) == 0:
				del(self.allRects[i])

		if self.unsavedChanges:
			print("saving JSON")
			jsonStr = json.dumps(self.allRects)
			with open(os.path.join(self.videosPath, self.jsonName), 'w') as fileHandle:
				fileHandle.write(str(jsonStr))
				fileHandle.close()
		else:
			print("no changes made")


		self.numVidAnnotations = len(list(self.allRects.keys()))
		self.numVidRects = 0
		for i in self.allRects:
			self.numVidRects += len(self.allRects[i])
		updateVidInfo = False
		if self.vidName in self.vidInfo and self.vidInfo[self.vidName][0] == self.numVidAnnotations and self.vidInfo[self.vidName][1] == self.numVidRects:
			pass
		else:
			self.vidInfo[self.vidName] = [self.numVidAnnotations, self.numVidRects]
			jsonStr = json.dumps(self.vidInfo)
			with open(os.path.join(self.videosPath, self.infoJsonName), 'w') as fileHandle:
				print("saving number of videos")
				fileHandle.write(str(jsonStr))
				fileHandle.close()


		self.unsavedChanges = False




if __name__ == "__main__":
	A = Annotator(videosPath, saveName, detector, numKeypoints)
	A.runThrough()