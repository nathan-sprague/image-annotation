import os
import cv2
import json
import numpy as np
import math
import time

saveName = ["earSlant"]
largerRectName = ""
modelName = ""

if saveName == ["ear"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/ear_videos/ear_rgb_training_videos"
	modelName = ""#"/home/nathan/Desktop/ear_stalk_detection/stalk-ear-detection/ear_finder/ear_tflite/corn_ear_oct16.tflite"

if saveName == ["earSlant"]:
	videosPath = "/home/nathan/Desktop/ear_stalk_detection/ear_videos/ear_rgb_training_videos"
	modelName = ""#"/home/nathan/Desktop/ear_stalk_detection/stalk-ear-detection/ear_finder/ear_tflite/corn_ear_oct16.tflite"


class Annotator:
	def __init__(self, path, labelName, modelName=""):
		self.currentRectangles = {"drawnRectangles": [], "tempRectangle": [[]], "possibleRectangles": [], "selectedRect": []}
		self.firstCallback = True
		self.path = path
		self.labelName = labelName
		self.fileLabelName = ""
		for i in self.labelName:
			self.fileLabelName += i + "_"
		self.fileLabelName = self.fileLabelName[0:-1]

		self.savableRects = {}
		self.unsavedChanges = False

		self.selectedRectInd = -1

		self.scale = 1
		self.mouseDown = False



		self.numRects = 0
		self.numImgs = 0

		self.labelingInd = 0

		self.makeAngles = False
		self.makingAngleLine = False
		self.minSectionStep = 5
		self.minSectionCount = 80
		self.sectionLabels = []

		self.makeSlant = True


		self.falseKey = -1


		self.defaultButtons = {"  delete": [(20, 20), (180, 80), 8], "    prev label": [(20, 100), (180, 160), 91], " prev": [(20, 180), (180, 240), 106], " next": [(20, 260), (180, 320), 108],
						"    next label": [(20, 340), (180, 400), 93], " export": [(20, 420), (180, 480), 202], " exit": [(20, 500), (180, 560), 27]} # poly used to be 116

		self.selectedButtons = {"  delete": [(20, 20), (180, 80), 8], "    deselect": [(20, 100), (180, 160), 199], " Trunc": [(20, 180), (180, 240), 200], "  Diff": [(20, 260), (180, 320), 201]}

		self.transformations = {"change color spaces":False, "mirror horizontally":False, "mirror vertically": False, "save just labeled rectangles": False, "Include Pascal VOC label (XML)": True}


		j = 0
		if len(self.labelName) > 1:
			for i in self.labelName:
				self.selectedButtons[i] = [(20, 340 + j*80), (180, 400 + j*80), j+300]
				j+=1

		self.buttons = self.defaultButtons

		self.sideBarWidth = 200

		self.useDetector = False
		if modelName != "":
			self.useDetector = True
			base_options = core.BaseOptions(
			  file_name=modelName, use_coral=False, num_threads=4)
			detection_options = processor.DetectionOptions(
				max_results=5, score_threshold=0.05)
			options = vision.ObjectDetectorOptions(
				base_options=base_options, detection_options=detection_options)
			self.detector = vision.ObjectDetector.create_from_options(options)



	def settingsEventHandler(self, event, x, y, flags, param):

		if event == 4 and x < 200 and y < 60:
			print("cancel")
			cv2.destroyWindow("export settings")
		elif event == 4 and x < 180 and x > 140:
			i=0
			for button in self.transformations:
				yMin = 70+50*i
				yMax = 110+50*i
				if yMin < y < yMax:
					self.transformations[button] = not self.transformations[button]
					self.showSettings()
					break
				i+=1
		elif event == 4:
			if 600 > x > 400 and 500 < y < 560:
				print("exporting!")
				cv2.destroyWindow("export settings")
				if self.transformations["save just labeled rectangles"]:
					self.saveImagesData()
				else:
					if self.makeSlant:
						self.saveNormalDataXML("yolo")
					else:
						self.saveNormalDataXML("xml")


	def showSettings(self):
		img = np.zeros((700, 1000,3), dtype=np.uint8)
		img[:,:] = (255, 255, 255)
		buttonsLabels = list(self.transformations.keys())

		self.countLabels()

		ogImgs = self.numImgs

		factor = 1
		# if self.transformations["rotate images"]:
		# 	factor *= 4
		if self.transformations["change color spaces"]:
			factor *= 5
		if self.transformations["mirror horizontally"]:
			factor *= 2
		if self.transformations["mirror vertically"]:
			factor *= 2
		newImgs = ogImgs * factor


		i=0
		while i < len(buttonsLabels):
			img = cv2.putText(img, buttonsLabels[i], (200, 100 + 50*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
			color = (0,0,255)
			if self.transformations[buttonsLabels[i]]:
				color = (0,255,0)
			img = cv2.rectangle(img, (140, 70+50*i), (180, 110+50*i), color, -1)
			i+=1


		img = cv2.putText(img, str(ogImgs) + " images turned to " + str(newImgs) + " after transformations", (200, 100 + 50*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

		img = cv2.rectangle(img, (0, 0), (200, 60), (250,255,0), 2)
		img = cv2.putText(img, "Cancel", (45, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

		img = cv2.rectangle(img, (400, 500), (600, 560), (250,255,0), 2)
		img = cv2.putText(img, "Export!", (445, 540), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)


		cv2.imshow("export settings", img)
		cv2.setMouseCallback("export settings", self.settingsEventHandler)



	def mouseEvent(self, event, x, y, flags, param):
		if x < self.sideBarWidth:
			if event == 4 and self.currentRectangles["tempRectangle"] == [[]]:
				self.mouseDown = False
				for b in self.buttons:
					if x>self.buttons[b][0][0] and x<self.buttons[b][1][0]:
						if y>self.buttons[b][0][1] and y<self.buttons[b][1][1]:
							print("buttoned")
							self.falseKey = self.buttons[b][2]
							return
				
			elif event == 1:
				return
			x = self.sideBarWidth


		x-=self.sideBarWidth
		self.sideBarWidth=max(0, self.sideBarWidth)
		if event == 6:
			self.falseKey = 108 # skip 5 frames if mouse wheel clicked


		x = int(x/self.scale)
		y = int(y/self.scale)


		self.processRectangleMouseEvent(event, x, y)



	def processRectangleMouseEvent(self, event, x, y):

		if event==1:
			self.mouseDown = True

			if self.makeAngles and self.selectedRectInd != -1:
				s = self.currentRectangles["selectedRect"][0]
				if s[0] < x < s[2] and s[1] < y < s[3]: 
					self.makingAngleLine = True
					self.currentRectangles["tempRectangle"][0] = [x,y,x,y]

					return

			self.selectedRectInd = -1
			self.currentRectangles["selectedRect"] = []


			if self.makeSlant and len(self.currentRectangles["tempRectangle"][0]) > 0:
				tr = self.currentRectangles["tempRectangle"][0]
				if tr[5] > 0:
					if (tr[3]-tr[5])**2 + (tr[4]-tr[6])**2 > 25:
					
						print("finished making rectangle")
						self.currentRectangles["drawnRectangles"] += [tr]
						self.currentRectangles["tempRectangle"][0] = []
						self.unsavedChanges = True
						self.numRects += 1
					else:
						print("rectangle too small")
						self.currentRectangles["tempRectangle"][0] = []
					self.dispImg()

			
			elif len(self.labelName) > 1:
				if self.makeSlant:
					self.currentRectangles["tempRectangle"][0] = [-2, x,y,x,y,-1,0,self.labelingInd]
				else:
					self.currentRectangles["tempRectangle"][0] = [x,y,x,y,0,0,self.labelingInd]
			else:
				if self.makeSlant:
					self.currentRectangles["tempRectangle"][0] = [-2, x,y,x,y,-1,0]
				else:
					self.currentRectangles["tempRectangle"][0] = [x,y,x,y,0,0]

		elif event==4:
			self.mouseDown = False

			tempRect = self.currentRectangles["tempRectangle"][0]
			if len(tempRect) > 0:



				if self.makingAngleLine and ( abs(tempRect[0] - tempRect[2]) > 5 or abs(tempRect[1]-tempRect[3]) > 5):
						extraZeros = [0]*(10-len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]))
						self.currentRectangles["drawnRectangles"][self.selectedRectInd] += extraZeros
						self.currentRectangles["drawnRectangles"][self.selectedRectInd][6] = tempRect[0]
						self.currentRectangles["drawnRectangles"][self.selectedRectInd][7] = tempRect[1]
						self.currentRectangles["drawnRectangles"][self.selectedRectInd][8] = tempRect[2]
						self.currentRectangles["drawnRectangles"][self.selectedRectInd][9] = tempRect[3]
						print("finished making arrow", self.currentRectangles["drawnRectangles"][self.selectedRectInd])
						self.unsavedChanges = True

				elif self.makeSlant and tempRect[5] == -1:
					print("starting next part")
					tr = self.currentRectangles["tempRectangle"][0]
					if (tr[1]-tr[3])**2 + (tr[2] - tr[4])**2 < 25:
						print("rect too small")
						self.currentRectangles["tempRectangle"] = [[]]
						self.checkDetection(tempRect)
					else:
						self.currentRectangles["tempRectangle"][0][5] = x
						self.currentRectangles["tempRectangle"][0][5] = y
					return

				elif abs(tempRect[0] - tempRect[2]) > 5 and abs(tempRect[1]-tempRect[3]) > 5:
					self.currentRectangles["drawnRectangles"] += [
					[min(tempRect[0], tempRect[2]), min(tempRect[1], tempRect[3]), max(tempRect[0], tempRect[2]), max(tempRect[1], tempRect[3]), 0, 0, self.labelingInd]]
					print("Made rectangle")
					self.numRects += 1
					self.unsavedChanges = True

				else:#if tempRect[0] == tempRect[2] and tempRect[1] == tempRect[3]:
					self.checkDetection(tempRect)

					if self.selectedRectInd == -1:
						print("didn't select anything")

				# else:
				# 	print("Rectangle too small")
				self.currentRectangles["tempRectangle"] = [[]]

				self.dispImg()

			else:
				print("no Rectangle to be made")


			self.makingAngleLine = False

			self.currentRectangles["tempRectangle"][0] = []

		elif event == 0 and self.mouseDown:
			h, w = self.openImg.shape[0:2]
			

			if len(self.currentRectangles["tempRectangle"][0]) > 1:
				if self.makeSlant:
					self.currentRectangles["tempRectangle"][0][3] = min(max(x,0),w)
					self.currentRectangles["tempRectangle"][0][4] = min(max(y,0),h)
				else:
					self.currentRectangles["tempRectangle"][0][2] = min(max(x,0),w)
					self.currentRectangles["tempRectangle"][0][3] = min(max(y,0),h)
			self.dispImg()
		elif event == 0 and self.makeSlant and len(self.currentRectangles["tempRectangle"][0]) > 0:
			tr = self.currentRectangles["tempRectangle"][0]
			if tr[5] > 0:
				angle = self.getAngle2pts((tr[1], tr[2]), (tr[3], tr[4]))
				if angle%(math.pi*2) < math.pi:
					if angle > math.pi*0.5:
						angle += math.pi/2
					else:
						angle -= math.pi/2
				else:
					if angle > math.pi*1.5:
						angle -= math.pi/2
					else:
						angle += math.pi/2
				if x > tr[3]:
					angle += math.pi/2
				else:
					angle -= math.pi/2
				d = math.sqrt((x-tr[3])**2 + (y-tr[4])**2)


				self.currentRectangles["tempRectangle"][0][5] = tr[3] + int(math.cos(angle)*d)
				self.currentRectangles["tempRectangle"][0][6] = tr[4] + int(math.sin(angle)*d)
				self.dispImg()

	def getAngle2pts(self, pt1, pt2):
		# returns in radians
		angle = 0
		if pt1[0] != pt2[0]:
			angle = (math.atan((pt1[1]-pt2[1])/(pt1[0]-pt2[0])))
			if pt2[0]<pt1[1]: # right
				angle += math.pi/2
			else: # left
				angle += math.pi*1.5
		elif pt2[1] > pt1[1]:
			angle = 0
		else:
			angle = math.pi
		return angle


	def checkDetection(self, tempRect):
		ind = 0
		print("checking")
		self.selectedRectInd = -1
		self.currentRectangles["selectedRect"] = [[]]

		for rect in self.currentRectangles["drawnRectangles"]:
			inside = False
			if rect[0] == -2:
				if tempRect[1] > min(rect[1],rect[3],rect[5]) and tempRect[1] < max(rect[1],rect[3],rect[5]):
					
					if tempRect[2] > min(rect[2],rect[4],rect[6]) and tempRect[2] < max(rect[2],rect[4],rect[6]):
						inside = True
			else:
				if tempRect[0] > rect[0] and tempRect[0] < rect[2]:
					if tempRect[1] > rect[1] and tempRect[1] < rect[3]:
						inside = True

			if inside:
				self.selectedRectInd = ind
				print("selected", self.selectedRectInd)
				self.currentRectangles["selectedRect"] = [rect]
				self.dispImg()
				break

			ind += 1

		if self.selectedRectInd == -1:
			ind = 0
			for rect in self.currentRectangles["possibleRectangles"]:
				if tempRect[0] > rect[0] and tempRect[0] < rect[2]:
					if tempRect[1] > rect[1] and tempRect[1] < rect[3]:
						self.selectedRectInd = len(self.currentRectangles["drawnRectangles"])
						self.currentRectangles["drawnRectangles"] += [rect]
						self.numRects += 1
						print("picked ML predicted")
						self.currentRectangles["selectedRect"] = [rect]
						break

				ind += 1
		if self.selectedRectInd == -1:
			print("didnt select anythign")
			self.dispImg()



	def detectPossible(self):
		rgb_image = cv2.cvtColor(self.openImg, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = self.detector.detect(input_tensor)
		possibleRects = []
		
		for i in detection_result.detections:
			x = i.bounding_box.origin_x
			y = i.bounding_box.origin_y
			w = i.bounding_box.width
			h = i.bounding_box.height
			if h > 3 and w > 3:
				possibleRects += [[x,y, x+w,y+h]]
		self.currentRectangles["possibleRectangles"] = possibleRects




	def dispImg(self):
		drawImg = self.openImg.copy()
		self.squareImg(self.openImg, self.currentRectangles["drawnRectangles"],500)
		# (720, 1280, 3)
		h, w = drawImg.shape[0:2]
		self.scale = 1

		if h/w > 720/1280:
			self.scale = 720/h
			drawImg = cv2.resize(drawImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)
		else:
			self.scale = 1280/w
			drawImg = cv2.resize(drawImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)


		for rectType in self.currentRectangles:
			thicknesses = {"drawnRectangles": 5, "tempRectangle": 1, "possibleRectangles": 1, "selectedRect": 5}
			colors = {"drawnRectangles": (255,0,0), "tempRectangle": (255,0,0), "possibleRectangles": (0,255,255), "selectedRect": (0,255,0)}


			for rect in self.currentRectangles[rectType]:
				if len(rect) > 0:
					if type(rect[0]) == int:

						if rect[0] == -2:
							if rect[5] > 0:
								lastPt = [(rect[1] + rect[5] - rect[3])* self.scale, (rect[2] + rect[6] - rect[4])* self.scale]
								pts = np.array([[rect[1]* self.scale, rect[2]* self.scale], [rect[3]* self.scale, rect[4]* self.scale], [rect[5]* self.scale, rect[6]* self.scale], lastPt], np.int32)


								drawImg = cv2.polylines(drawImg, [pts], True, colors[rectType], thicknesses[rectType])

							else:
								drawImg = cv2.line(drawImg, (int(rect[1]*self.scale), int(rect[2]*self.scale)), (int(rect[3]*self.scale), int(rect[4]*self.scale)), colors[rectType], thicknesses[rectType])
						

						else:
							if len(rect) > 5:
								if rect[4] == 1: # truncated
									drawImg = cv2.putText(drawImg, "t", (int(rect[0]*self.scale+40), int(rect[1]*self.scale-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
								if rect[5] == 1: # difficult
									drawImg = cv2.putText(drawImg, "d", (int(rect[0]*self.scale), int(rect[1]*self.scale-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

							if len(self.labelName) > 1:
								center = (int((rect[0]+rect[2])*self.scale/2), int((rect[1]+rect[3])*self.scale/2))
								if len(rect) > 6:

									drawImg = cv2.putText(drawImg, self.labelName[rect[6]][0], center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
								else:
									drawImg = cv2.putText(drawImg, self.labelName[0][0], center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

							if len(rect) > 8 and self.makeAngles:
								angle = self.getAngle(rect)

								center = (int((rect[0]+rect[2])*self.scale/2), int((rect[1]+rect[3])*self.scale/2))
								drawImg = cv2.putText(drawImg, str(angle), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

								drawImg = cv2.arrowedLine(drawImg, (int(rect[6]*self.scale), int(rect[7]*self.scale)), (int(rect[8]*self.scale), int(rect[9]*self.scale)), colors[rectType], thicknesses[rectType])
						


							if self.makingAngleLine and rectType == "tempRectangle":
								drawImg = cv2.line(drawImg, (int(rect[0]*self.scale), int(rect[1]*self.scale)), (int(rect[2]*self.scale), int(rect[3]*self.scale)), colors[rectType], thicknesses[rectType])

							else: 
								drawImg = cv2.rectangle(drawImg, (int(rect[0]*self.scale), int(rect[1]*self.scale)), (int(rect[2]*self.scale), int(rect[3]*self.scale)), colors[rectType], thicknesses[rectType])
						

					else:

						furthest = [-1, 0]
						pts = []
						for i in rect:
							pts += [(int(i[0]*self.scale), int(i[1]*self.scale))]
							if (pts[0][0]-pts[-1][0])**2 + (pts[0][1]-pts[-1][1])**2 > furthest[0]:
								furthest = [(pts[0][0]-pts[-1][0])**2 + (pts[0][1]-pts[-1][1])**2, pts[-1]]

						# if (pts[0][0]-furthest[1][0]) != 0:
						# 	print("angle", math.degrees(math.atan((pts[0][1]-furthest[1][1]) / (pts[0][0]-furthest[1][0]) )))
						pts = np.array(pts, np.int32)
			

						color = colors[rectType]
						thick = thicknesses[rectType]
						if rectType == "tempRectangle":
							drawImg = cv2.circle(drawImg, pts[0], 5, (0,0,255), 2)
							closed = False
							if abs(pts[-1][0]-pts[0][0]) < 5 and abs(pts[-1][1]-pts[0][1]) < 5:
								color = (0,255,0)
								thick = 2
								closed = True
						else:
							closed = True
							drawImg = cv2.line(drawImg, pts[0], furthest[1], (0,0,255), 2)



						drawImg = cv2.polylines(drawImg, [pts], closed, color, thick)




		h, w = drawImg.shape[0:2]
		zeros = np.zeros((h,w+self.sideBarWidth,3), dtype=np.uint8)
		zeros[:, self.sideBarWidth::] = drawImg
		drawImg = zeros

		if self.selectedRectInd == -1:
			self.buttons = self.defaultButtons
			for b in self.buttons:
				button = self.buttons[b]
				drawImg = cv2.rectangle(drawImg, button[0], button[1], (255,255,255), -1)
				drawImg = cv2.rectangle(drawImg, button[0], button[1], (255,255,0), 3)
				center = (int((button[0][0]+button[1][0])/2-10*len(b)), int((button[0][1]+button[1][1])/2)+8)
				drawImg = cv2.putText(drawImg, b, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

			dirText = "file " + str(self.dirInd+1) + "/" + str(len(self.dirs))
			drawImg = cv2.putText(drawImg, dirText, (20, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

			dirText = "frame " + str(self.frameNum) + "/" + str(self.maxFrameNum)
			drawImg = cv2.putText(drawImg, dirText, (15, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

			dirText = str(self.numRects) + " labels"
			drawImg = cv2.putText(drawImg, dirText, (15, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

			dirText = str(self.numImgs) + " imgs"
			drawImg = cv2.putText(drawImg, dirText, (15, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
		else:
			self.buttons = self.selectedButtons
			for b in self.buttons:
				button = self.selectedButtons[b]
				drawImg = cv2.rectangle(drawImg, button[0], button[1], (255,255,255), -1)
				drawImg = cv2.rectangle(drawImg, button[0], button[1], (255,255,0), 3)
				center = (int((button[0][0]+button[1][0])/2-10*len(b)), int((button[0][1]+button[1][1])/2)+8)
				drawImg = cv2.putText(drawImg, b, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

			dirText = "rect " + str(self.selectedRectInd+1)
			drawImg = cv2.putText(drawImg, dirText, (20, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

			truncated = 0
			difficult = 0
			if len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]) > 5:
				truncated = self.currentRectangles["drawnRectangles"][self.selectedRectInd][4]
				difficult = self.currentRectangles["drawnRectangles"][self.selectedRectInd][5]
			dirText = "truncated: " + str(truncated)
			drawImg = cv2.putText(drawImg, dirText, (20, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

			dirText = "difficult: " + str(difficult)
			drawImg = cv2.putText(drawImg, dirText, (20, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)


		cv2.imshow("img", drawImg)


		if self.firstCallback:
			cv2.setMouseCallback("img", self.mouseEvent)

			self.firstCallback = False

	def pointInPoly(self, x,y, poly):
		# taken from https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
	    n = len(poly)
	    inside = False

	    p1x,p1y = poly[0]
	    for i in range(n+1):
	        p2x,p2y = poly[i % n]
	        if y > min(p1y,p2y):
	            if y <= max(p1y,p2y):
	                if x <= max(p1x,p2x):
	                    if p1y != p2y:
	                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
	                    if p1x == p2x or x <= xints:
	                        inside = not inside
	        p1x,p1y = p2x,p2y

	    return inside


	def manageKeyResponses(self):
		jumpAmount = 0
		while jumpAmount == 0:

			self.falseKey = -1
			k = -1
			while self.falseKey == -1 and k == -1:
				k = cv2.waitKey(10)

			if self.falseKey != -1:
				k =  self.falseKey

			print(k)

			if k == 27: # esc
				if self.mouseDown:
					self.mouseDown = False
					self.currentRectangles["tempRectangle"] = [[]]
					self.dispImg()
				else:
					self.saveDataJSON()
					print("done")

					exit()

			elif k == 108: # l
				if self.fileType == "img":
					jumpAmount = 1
				else:
					jumpAmount = 5

			elif k == 106: # j
				if self.fileType == "img":
					jumpAmount = -1
				else:
					jumpAmount = -5

			elif k == 47: # / skip to next video
				jumpAmount = self.maxFrameNum + 2

			elif k == 46: # . skip to last video
				jumpAmount = -self.maxFrameNum - 2

			elif k == 91: # [ skip to closest smaller identified rectangle
				framesToGo = np.array([eval(i) for i in list(self.savableRects.keys())])
				if len(framesToGo) > 0:
					if self.frameNum > framesToGo.min():
						closest = framesToGo[framesToGo < self.frameNum].max()
						jumpAmount = closest - self.frameNum
					else:
						print("unable to skip, none before")
				else:
					print("no identified rectangles")


			elif k == 93: # ] skip to closest larger identified rectangle
				framesToGo = np.array([eval(i) for i in list(self.savableRects.keys())])
				if len(framesToGo) > 0:
					if self.frameNum < framesToGo.max():
						closest = framesToGo[framesToGo > self.frameNum].min()
						jumpAmount = closest - self.frameNum
					else:
						print("unable to skip, none above")
				else:
					print("no identified rectangles")

			elif k == 92: # \ skip to last identified rectangle
				framesToGo = np.array([eval(i) for i in list(self.savableRects.keys())])
				if len(framesToGo) > 0:
					jumpAmount = framesToGo.max() - self.frameNum
				else:
					print("no identified rectangles")

			elif k == 115: # s
				print("Saving JSON")
				self.saveDataJSON()

			elif k == 100:
				if self.selectedRectInd != -1:
					if len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]) < 6:
						self.currentRectangles["drawnRectangles"][self.selectedRectInd] += [0,0]
					self.currentRectangles["drawnRectangles"][self.selectedRectInd][5] = (self.currentRectangles["drawnRectangles"][self.selectedRectInd][5] + 1) % 2
					self.dispImg()
				else:
					print("no rectangle selected")

			elif k == 116:
				if self.selectedRectInd != -1:
					if len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]) < 6:
						self.currentRectangles["drawnRectangles"][self.selectedRectInd] += [0,0]
					self.currentRectangles["drawnRectangles"][self.selectedRectInd][4] = (self.currentRectangles["drawnRectangles"][self.selectedRectInd][4] + 1) % 2
					self.dispImg()
				else:
					print("no rectangle selected")

			elif k == 112: # p
				print("exporting images")
				
				self.saveNormalDataXML()

			elif k == 111: # o
				print("exporting just images")
				self.saveImagesData()

			elif k == 8: # backspace

				print("delete", self.selectedRectInd)
				if self.selectedRectInd >= 0:
					self.currentRectangles["drawnRectangles"] = self.currentRectangles["drawnRectangles"][0:self.selectedRectInd] + self.currentRectangles["drawnRectangles"][self.selectedRectInd+1::]
					print("deleted selected rectangle")
					self.currentRectangles["selectedRect"] = []
					self.selectedRectInd = -1
					self.unsavedChanges = True
					self.numRects -= 1
					self.dispImg()


				elif len(self.currentRectangles["drawnRectangles"]) > 0:
					self.currentRectangles["drawnRectangles"] = self.currentRectangles["drawnRectangles"][0:-1]
					print("deleted last rectangle")
					self.unsavedChanges = True
					self.numRects -= 1
					self.dispImg()

				else:
					print("no rectangles to delete")

			elif k == 99: # c
				print("clear")
				self.unsavedChanges = True
				self.currentRectangles = {"drawnRectangles": [], "tempRectangle": [[]], "possibleRectangles": [], "selectedRect": []}
				self.dispImg()
			elif k == 122:
				jumpAmount = 99999

			elif k == 199: # deselect
				self.selectedRectInd = -1
				self.currentRectangles["selectedRect"] = []
				self.dispImg()
				print("deselect")

			elif k == 200: # toggle truncated
				print("drawnRectangles", self.currentRectangles["drawnRectangles"][self.selectedRectInd])
				if len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]) == 4:
					self.currentRectangles["drawnRectangles"][self.selectedRectInd] += [0,0]
				self.currentRectangles["drawnRectangles"][self.selectedRectInd][4] = (self.currentRectangles["drawnRectangles"][self.selectedRectInd][4] + 1) % 2
				self.unsavedChanges = True
				self.dispImg()
				print("toggled truncated")

			elif k == 201: # toggle difficulty
				if len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]) == 4:
					self.currentRectangles["drawnRectangles"][self.selectedRectInd] += [0,0]
				self.currentRectangles["drawnRectangles"][self.selectedRectInd][5] = (self.currentRectangles["drawnRectangles"][self.selectedRectInd][5] + 1) % 2
				self.unsavedChanges = True
				self.dispImg()
				print("toggled difficulty")
			elif k == 202:
				self.showSettings()
			elif 300 <= k < 300 + len(self.labelName):
				if len(self.currentRectangles["drawnRectangles"][self.selectedRectInd]) < 6:
					zerosToAdd = 7-len(self.currentRectangles["drawnRectangles"][self.selectedRectInd])
					self.currentRectangles["drawnRectangles"][self.selectedRectInd] += [0] * zerosToAdd
					print("added",zerosToAdd,"zeros", self.currentRectangles["drawnRectangles"][self.selectedRectInd])
					self.unsavedChanges = True
				self.currentRectangles["drawnRectangles"][self.selectedRectInd][6] = k-300
				self.labelingInd = k - 300
				self.dispImg()


			else:
				jumpAmount += 1


		rects = self.currentRectangles["drawnRectangles"]
		self.currentRectangles = {"drawnRectangles": [], "tempRectangle": [[]], "possibleRectangles": [], "selectedRect": []}
		print(len(rects), "labels made")
		self.selectedRectInd = -1
		return jumpAmount, rects





	def goThroughDir(self):
		print("starting")
		self.allFiles = os.listdir(self.path)
		self.dirs = []
		for i in self.allFiles:
			if i[-4::] in [".avi", ".png", ".jpg", ".mov", ".MOV"]:
				self.dirs += [i]
		self.countLabels()

		self.dirs.sort()

		self.dirInd = 0

		self.prelabeledInd = 0

		self.frameNumInd = 0
		self.desiredFrameNum = 0
		self.maxFrameNum = 0


		while self.dirInd < len(self.dirs):
			self.desiredFrameNum = 0

			fileName = self.dirs[self.dirInd]
			print("reading", fileName, "file number", self.dirInd+1, "out of", len(self.dirs))

			self.jsonName = fileName[0:-4] + self.fileLabelName + ".json"
			if self.jsonName in self.allFiles:
				with open(self.path + '/' + self.jsonName) as json_file:
					data = json.load(json_file)
				print("loaded in", data)
				self.savableRects = data
			else:
				self.savableRects = {}
				print("no annotation data called", self.jsonName)
			self.unsavedChanges = False



			if fileName[-4::] in [".avi", ".mov", ".MOV"]:
				self.fileType = "vid"
				cap = cv2.VideoCapture(self.path + "/" + fileName)
				self.maxFrameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
				if self.maxFrameNum == 0:
					print("number of frames not known. May need to move through manually")
					self.maxFrameNum = 0
					

				self.frameNum = 0
				while cap.isOpened():

					
					cap.set(1,self.desiredFrameNum)

					ret, self.openImg = cap.read()


					

					if not ret:
						print("failed to read frame")
						self.dirInd += 1
						break

					oldFrameNum = self.frameNum
					self.frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
					print(oldFrameNum, self.frameNum)
					if oldFrameNum == self.frameNum:
						print("the video is strange, attempting to fix")
						cap.release()
						self.fixVideo(self.path + "/" + fileName)
						cap = cv2.VideoCapture(self.path + "/" + fileName)
						cap.set(1,self.desiredFrameNum)
						ret, self.openImg = cap.read()
						self.frameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
						self.maxFrameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

						print("back in business!")
					

					print("reading frame", self.frameNum, "out of", self.maxFrameNum)


					if str(self.frameNum) in self.savableRects:
						self.currentRectangles["drawnRectangles"] = self.savableRects[str(self.frameNum)][:]
					elif self.useDetector:
						self.detectPossible()

					self.dispImg()
					self.getTransformationTypes()
				
					change, rectangles = self.manageKeyResponses()

					if len(rectangles) > 0:
						self.savableRects[str(self.frameNum)] = rectangles
					elif str(self.frameNum) in self.savableRects:
						self.savableRects.pop(str(self.frameNum))

					self.desiredFrameNum += change

					if change == 99999:
						print("skippping")
						while self.dirInd < len(self.dirs):
							self.dirInd +=1
							fileName = self.dirs[self.dirInd]
							self.jsonName = fileName[0:-4] + self.fileLabelName + ".json"
							if self.jsonName not in self.allFiles:
								break
							print("dir ind",self.dirInd)
						break

					if self.desiredFrameNum >= self.maxFrameNum and self.maxFrameNum != 0:
						print("video ended")
						if self.dirInd < len(self.dirs):
							self.dirInd += 1
						break
					elif self.desiredFrameNum < 0:
						print("previous video")
						if self.dirInd > 0:
							self.dirInd -= 1

						break

				cap.release()




			elif fileName[-4::] in [".png", ".jpg"]:
				self.fileType = "img"
			
				self.openImg = cv2.imread(self.path + "/" + fileName)


				self.frameNum = 1
				if str(1) in self.savableRects:
					self.currentRectangles["drawnRectangles"] = self.savableRects[str(1)][:]
				elif self.useDetector:
					self.detectPossible()

				print("displaying image")
				self.dispImg()


				change, rectangles = self.manageKeyResponses()

				if change == 99999:
					print("skippping")
					while self.dirInd < len(self.dirs):
						self.dirInd +=1
						fileName = self.dirs[self.dirInd]
						self.jsonName = fileName[0:-4] + self.fileLabelName + ".json"
						if self.jsonName not in self.allFiles:
							break
						print("dir ind",self.dirInd)
					print("going to", self.dirInd)
				elif change == -99999:
					print("skipping back")
					while self.dirInd > 0:
						self.dirInd -= 1
						fileName = self.dirs[self.dirInd]
						self.jsonName = fileName[0:-4] + self.fileLabelName + ".json"
						if self.jsonName in self.allFiles:
							break
						print("dir ind",self.dirInd)
					print("going to", self.dirInd)

				else:
					self.dirInd += change


				if rectangles == []:
					self.savableRects = {}
					self.unsavedChanges = True
				else:
					self.savableRects["1"] = rectangles


				
				if self.dirInd < 0:
					print("cant go back more")
					self.dirInd = 0

			else:
				print("unknown file type")
				self.dirInd += 1

			self.saveDataJSON()

			if self.dirInd == len(self.dirs):
				print("no more videos!")
				self.dirInd -= 1


		img = np.zeros((300,300,3), dtype=np.uint8)

		cv2.imshow("img", img)
		k = 0
		while k != 27:
			k = cv2.waitKey(0)
			print("press escape")
		print("done")


	def saveDataJSON(self):
		if self.unsavedChanges:
			if self.savableRects != {}:
				jsonStr = json.dumps(self.savableRects)
				with open(self.path + "/" + self.jsonName, 'w') as fileHandle:
					fileHandle.write(str(jsonStr))
					fileHandle.close()

				print("Saving", jsonStr)
				print("saved", self.jsonName)
				if self.jsonName not in self.allFiles:
					self.allFiles += [self.jsonName]

			elif self.jsonName in self.allFiles:
				os.remove(self.path + "/" + self.jsonName)
				self.allFiles.remove(self.jsonName)
				print("had saved something, nothing left so deleted it")
			else:
				print("nothing was saved, nothing to save")
		else:
			print("no changes to save")
		self.unsavedChanges = False


	def saveImagesData(self):
		# save just the cropped images
		dirName = self.path + "/" + self.fileLabelName + "_images"
		if not os.path.exists(dirName):
			os.makedirs(dirName)

		rectsSaved = 0
		for fileName in self.dirs:
			jsonName = fileName[0:-4] + self.fileLabelName + ".json"
			if jsonName in self.allFiles:

				with open(self.path + '/' + jsonName) as json_file:
					data = json.load(json_file)
				print("found data for", fileName)
				frameNums = [eval(i) for i in list(data.keys())]
				cap = cv2.VideoCapture(self.path + "/" + fileName)

				for frameNum in frameNums:
					cap.set(1,frameNum-1)
					ret, img = cap.read()
					if not ret:
						print("no frame found at frame number", frameNum)
						break

					
					i = 0
					for rectNum in data[str(frameNum)]:
						i+=1
						if rectNum[2] > rectNum[0] and rectNum[3] > rectNum[1]:
							im = img[rectNum[1]:rectNum[3], rectNum[0]:rectNum[2]]

							h, w = im.shape[0:2]
							if h > 0 and w > 0:
								scale = 1
								if h < 300 and w < 300:
									scale = max(400/h, 400/w)
								im = cv2.resize(im, (int(scale * w), int(scale*h)), interpolation = cv2.INTER_AREA)
								saveName = fileName[0:-4] + "_f" + str(frameNum) + "_r" + str(i) + ".jpg"

								# cv2.imwrite(dirName + "/" + saveName, im)
								if True:
									cv2.imshow("saving", im)
									cv2.waitKey(1)
								rectsSaved += 1

		cv2.destroyWindow('saving')
		print("done saving", rectsSaved, "rectangles saved")


	def transformImg(self, img, rects, transformations):

		rectangles = []
		for rect in rects:
			rectangles += [rect[:]] # do a deep copy

		h,w = img.shape[0:2]
		res = img.copy()
		# res = cv2.resize(res, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)
		# i=0
		# while i < transformations[0]:
		# 	res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
		# 	# if i == 0:
		# 	h,w = img.shape[0:2]
		# 	for r in rectangles:
		# 		r[0:4] = [w/2 - (r[1] - h/2), h/2 + (r[0] - w/2),   w/2 - (r[3] - h/2), h/2 + (r[2] - w/2)]
		# 	i+=1
		

		brightnessFactor = -20
		if transformations[1] * brightnessFactor > 0:
			res[res >= 255 - transformations[1] * brightnessFactor] = 255
			res[res < 255] += transformations[1] * brightnessFactor
		elif transformations[1] * brightnessFactor < 0:
			res[res <= -transformations[1] * brightnessFactor] = 0
			res[res > 0] -= -transformations[1] * brightnessFactor


		if transformations[2]==1: # horizontal
			res = cv2.flip(res, 1)
			for r in rectangles:
				r[0:4] = [w-r[2], r[1], w-r[0], r[3]]

		if transformations[3]==1:
			res = cv2.flip(res, 0)
			for r in rectangles:
				r[0:4] = [r[0], h-r[3], r[2], h-r[1]]


		# cv2.imshow("Res", res)
		# cv2.waitKey(100)
		return res, rectangles



	def getTransformationTypes(self):
		# self.transformations = {"rotate images":True, "change color spaces":True, "mirror horizontally":True, "mirror vertically": True, "save just labeled rectangles": True, "Include Pascal VOC label (XML)": True}
		transformations = [[0,0,0,0]]

		# if self.transformations["rotate images"] and False: # not working right now
		# 	listToAdd = []
		# 	for sublist in transformations:
		# 		j = 1
		# 		while j < 4:
		# 			listToAdd += [sublist[:]]
		# 			listToAdd[-1][0] = j
		# 			j+=1
		# 	transformations += listToAdd
		# 	print("added", len(listToAdd))

		if self.transformations["change color spaces"]:
			listToAdd = []
			for sublist in transformations:
				brightnesses = (-2,-1,1,2)
				for j in brightnesses:
					listToAdd += [sublist[:]]
					listToAdd[-1][1] = j
			transformations += listToAdd
			print("added", len(listToAdd))

		if self.transformations["mirror horizontally"]:
			listToAdd = []
			for sublist in transformations:
				listToAdd += [sublist[:]]
				listToAdd[-1][2] = 1
			transformations += listToAdd
			print("added", len(listToAdd))

		if self.transformations["mirror vertically"]:
			listToAdd = []
			for sublist in transformations:
				listToAdd += [sublist[:]]
				listToAdd[-1][3] = 1
			transformations += listToAdd
			print("added", len(listToAdd))


		return transformations






	def saveNormalDataXML(self, saveFormat="XML"):
		self.saveDataJSON()

		if saveFormat == "yolo": # 100 validate, 1500 train, 70 test
			print("saving yolo files")
			yamlTxt = "path: \n\n"
			dirNames = ["train", "valid", "test"]
			i=0
			mainFolder = self.path + "/yolo_"+self.fileLabelName
			if os.path.exists(mainFolder):
				os.rmdir(mainFolder)
			os.makedirs(mainFolder)

			while i < len(dirNames):
				yamlTxt += dirNames[i] + ": " + mainFolder + "/" + dirNames[i] + "/images\n"
				dirNames[i] = mainFolder + "/" + dirNames[i]
				os.makedirs(dirNames[i])
				os.makedirs(dirNames[i] + "/" + "images")
				os.makedirs(dirNames[i] + "/" + "labelTxt")


				i+=1
			yamlTxt += "\nnc: " + str(len(self.labelName)) + "\n" + "names: " + str(self.labelName)


			with open( mainFolder + "/data.yaml", 'w') as fileHandle:
				fileHandle.write(yamlTxt)
				fileHandle.close()

		else:


			print("saving xml files")
			trainDirName = self.path + "/" + self.fileLabelName + "_train"
			if not os.path.exists(trainDirName):
				os.makedirs(trainDirName)
			validDirName = self.path + "/" + self.fileLabelName + "_validate"
			if not os.path.exists(validDirName):
				os.makedirs(validDirName)

		showImgs = True
		imgCount = 0
		labelCount = 0
		missed = 0
		transformations = self.getTransformationTypes()
		for fileName in self.dirs:

			jsonName = fileName[0:-4] + self.fileLabelName + ".json"
			if jsonName in self.allFiles:

				with open(self.path + '/' + jsonName) as json_file:
					data = json.load(json_file)
				print("found data for", fileName)

				if fileName[-4::] in [".mov", ".MOV", ".avi"]:
					frameNums = [eval(i) for i in list(data.keys())]
					cap = cv2.VideoCapture(self.path + "/" + fileName)

					for frameNum in frameNums:
						cap.set(1,frameNum-1)
						ret, img = cap.read()


						rects = data[str(frameNum)]

						if saveFormat == "yolo":
							img, rects = self.squareImg(img, rects, 500)
							if rects == False:
								print("not using")
								missed += 1
								continue

							if imgCount%10 == 0:
								dirName = dirNames[1] # validate
							elif imgCount%11 == 0:
								dirName = dirNames[2] # test
							else:
								dirName = dirNames[0] # train



						else:
							if imgCount%10 == 0:
								dirName = validDirName
							else:
								dirName = trainDirName
						
						saveName = fileName[0:-4] + "_f" + str(frameNum) + ".jpg"

						imgCount += 1

						
						

						
						if not ret:
							print("no frame found at frame number", frameNum)
							break

						if showImgs:
							if imgCount%10 == 0:
								cv2.imshow("saving", img)
								cv2.waitKey(1)


						
						labelCount += len(rects)

						saveTypes = []
						skip = False
						if self.makeAngles:
							for i in rects:
								angle = self.getAngle(i)
								if angle == False:
									print("no angle found for frame, skipping.")
									skip = True
									break
								j=0
								while j < len(self.sectionLabels):
									if self.sectionLabels[j][0]<=angle<self.sectionLabels[j][1]:
										saveTypes += [str(self.sectionLabels[j][0]) + "_" + str(self.sectionLabels[j][1])]
										break
									j+=1
								if j == len(self.sectionLabels):
									print("Section label never added for angle", angle)
						else:	
							for i in rects:
								if len(i) > 6 and not self.makeSlant:
									saveTypes += [self.labelName[i[6]]]
								else:
									saveTypes += [self.labelName[0]]

						if not skip:

							tNum = 0
							for t in transformations:
								sn = saveName[0:-4] + "_" + str(tNum) + saveName[-4::]
								imWrite, r = self.transformImg(img, rects, t)
								if saveFormat == "yolo":
									cv2.imwrite(dirName + "/images/" + sn, imWrite)
								else:
									cv2.imwrite(dirName + "/" + sn, imWrite)

								if self.transformations["Include Pascal VOC label (XML)"]:
									if saveFormat == "yolo":
										self.saveYOLO(dirName+"/labelTxt/" + sn[0:-4] + ".txt", r, saveTypes)
									else:
										self.saveXML(dirName,  sn, r, saveTypes, imWrite.shape[0:2])
								tNum += 1

				elif fileName[-4::] in [".jpg", ".png"]:

					if "1" in data:

						if imgCount%10 == 0:
							dirName = validDirName
						else:
							dirName = trainDirName
						imgCount += 1

						print(data["1"])
						rects = data["1"]
						saveName = fileName

						saveTypes = []

						if self.makeAngles:
							for i in rects:
								angle = self.getAngle(i)
								saveTypes += [self.sectionLabels[self.getAngleCategory(angle)]]
						else:
							for i in rects:
								if len(i) > 6:
									saveTypes += [self.labelName[i[6]]]
								else:
									saveTypes += self.labelName[0]

						img = cv2.imread(self.path + "/" + fileName)
						cv2.imshow("saving", img)
						cv2.waitKey(1)
						labelCount += len(rects)

						tNum = 0
						for t in transformations:
							sn = saveName[0:-4] + "_" + str(tNum) + saveName[-4::]
							imWrite, r = self.transformImg(img, rects, t)
							cv2.imwrite(dirName + "/" + sn, imWrite)
							if self.transformations["Include Pascal VOC label (XML)"]:
								self.saveXML(dirName,  sn, r, saveTypes, imWrite.shape[0:2])
							tNum += 1
				else:
					print("unknown file", fileName, "type", fileName)
			


		if showImgs:
			cv2.destroyWindow('saving')
			
		print("saved", labelCount, "labels from", imgCount, "frames. Missed", missed)


		print("Saved all")


	def squareImg(self, img, rects, finalSize):
		if len(rects) == 0:
			return False, False
		# for only slanted rects. may modify for all rects later
		h, w = img.shape[0:2]
		smallestSizes = [w, h]
		largestSizes = [0, 0]
		for r in rects:
			i=1
			while i<len(r):
				smallestSizes[0] = min(r[i], smallestSizes[0])
				smallestSizes[1] = min(r[i+1], smallestSizes[1])
				largestSizes[0] = max(r[i], largestSizes[0])
				largestSizes[1] = max(r[i+1], largestSizes[1])
				i+=2
		smallestSquare = (abs(largestSizes[0]-smallestSizes[0]), abs(largestSizes[1]-smallestSizes[1]))
		if max(smallestSquare) > min(h, w):
			print("too hard to shrink. smallest square", smallestSquare, (h,w))
			return False, False
		else:
			cropSize = min(h, w)
			center = [(smallestSizes[0] + largestSizes[0])/2, (smallestSizes[1] + largestSizes[1])/2]
			startFromBack=[center[0]-cropSize/2, center[1]-cropSize/2]
			if startFromBack[0] < 0:
				startFromBack[0] = 0
			if startFromBack[1] < 0:
				startFromBack[1] = 0	
			if startFromBack[0]+cropSize > w:
				startFromBack[0] = w-cropSize
			if startFromBack[1]+cropSize > h:
				startFromBack[1] = h-cropSize
			startFromBack = [int(startFromBack[0]),  int(startFromBack[1])]


			crop = img.copy()
			crop = crop[startFromBack[1]:cropSize+startFromBack[1], startFromBack[0]:cropSize+startFromBack[0]]
			scale = finalSize/cropSize
			crop = cv2.resize(crop, (finalSize, finalSize), interpolation = cv2.INTER_AREA)
			# scale = 1

			cv2.line(img, (0, startFromBack[1]), (w, startFromBack[1]), (0,0,0), 2)
			cv2.line(img, (startFromBack[0], 0), (startFromBack[0], h), (0,0,0), 2)
			newRects = []
			for i in rects:
				rect = [0,0,0,0,0,0]
				j=0
				while j < 6:
					rect[j] = (i[j+1]-startFromBack[(j)%2])*scale
					j+=1
				newRects += [[0] + rect]
				# pts = np.array([[rect[0], rect[1]], [rect[2], rect[3]], [rect[4], rect[5]]], np.int32)
				# crop = cv2.polylines(crop, [pts], True, (255,255,0), 3)

			# cv2.imshow("cropped", crop)

			return crop, newRects








	def getAngle(self, rect):
		if len(rect) < 9:
			return False
		# get the angle assuming rect is [x, x, x, x, x, (x1), (y1), (x2), (y2)]. Returns in degrees
		angle = 0
		if rect[6] != rect[8]:
			angle = math.degrees(math.atan((rect[9]-rect[7])/(rect[8]-rect[6])))
			if rect[6]<rect[8]: # right
				angle += 90
			else: # left
				angle += 270
		elif rect[7] > rect[9]:
			angle = 0
		else:
			angle = 180
		return int(angle)

	def countLabels(self):
		imgs = 0
		labels = 0
		numVideos = 0
		allAngles = []
		allAnglesImg = []
		for fileName in self.dirs:
			jsonName = fileName[0:-4] + self.fileLabelName + ".json"
			if jsonName in self.allFiles:
				numVideos += 1
				with open(self.path + '/' + jsonName) as json_file:
					data = json.load(json_file)
					imgs += len(data.keys())
					for frameNum in data:
						allAnglesImg += [[]]
						labels += len(data[frameNum])
						if self.makeAngles:
							for rect in data[frameNum]:
								if len(rect) > 8:
									allAngles += [self.getAngle(rect)]
									allAnglesImg[-1] += [self.getAngle(rect)]
								else:
									print("unlabeled rectangle found on frame", frameNum, "of video", fileName)
		self.numRects = labels
		self.numImgs = imgs

		
		if self.makeAngles:
			sections = [0] * 360

			for angle in allAngles:
				sections[angle] +=1

			countInSection = 0
			angle = 0
			startAngle = 0
			self.sectionLabels = []
			sectionAmounts = []
			for i in sections:
				
				if countInSection > self.minSectionCount and angle - startAngle > self.minSectionStep:
					self.sectionLabels += [[startAngle, angle]]
					sectionAmounts += [countInSection]
					print(sections[startAngle:angle])
					countInSection = 0
					startAngle = angle
				countInSection += i
				angle += 1
			self.sectionLabels[-1][1] = 360

			print("new labels", self.sectionLabels)
			print("section amounts", sectionAmounts)

			textLabels = []
			for i in self.sectionLabels:
				textLabels += [str(i[0]) + "_" + str(i[1])]
			print("labels as text", textLabels)

			print("section distribution", sections)

			imgsPerLabel = [0]*len(self.sectionLabels)
			n=0
			for imgAngles in allAnglesImg:

				labelsGiven = set()
				for angle in imgAngles:
					gaveLabel = False
					i=0
					for j in self.sectionLabels:
						if j[0] <= angle < j[1]:
							labelsGiven.add(i)
							gaveLabel = True
							break
						i+=1
					
					if not gaveLabel:
						print("uh oh- problem !")
						return
				print("given", labelsGiven, "from", imgAngles)
			

				for i in labelsGiven:
					imgsPerLabel[i] += 1
			print("imgs per label", imgsPerLabel)




	def getAngleCategory(self, angle):
			
		if angle >= 360-self.sectionStep/2 or angle < self.sectionStep/2:
			return 0
			# print("angle", angle, "in section", sectionLabels[0])
		else:
			res = angle - self.sectionStep/2
			res = int(res/self.sectionStep)+1
			return res
			# print("angle", angle, "in section", sectionLabels[res])
	


	def saveYOLO(self, fileName, rectangles, saveTypes):
		text = ""
		i=0
		while i < len(rectangles):
			dims = rectangles[i][1:7]
			dx = dims[4]-dims[2]
			dy = dims[5]-dims[3]
			dims += [dims[4]+dx, dims[5]+dy]
			for j in dims:
				text+= str(j) + " "
			text += saveTypes[i] + " 0\n"
			i+=1
		with open(fileName, 'w') as fileHandle:
			fileHandle.write(text)
			fileHandle.close()



	def saveXML(self, folder, fileName, rectangles, saveTypes, imshape):

		text = """
<annotation>
\t<folder>""" + folder + """</folder>
\t<filename>""" + fileName[0:-4] + ".jpg" + """</filename>
\t<path>""" + folder + "/"  + fileName[0:-4] + ".jpg" + """</path>
\t<source>
\t\t<database>Unknown</database>
\t</source>
\t<size>
\t\t<width>""" + str(imshape[1]) + """</width>
\t\t<height>""" + str(imshape[0]) + """</height>
\t\t<depth>3</depth>
\t</size>
\t<segmented>0</segmented>\n"""

		i=0
		while i < len(rectangles):
			l = rectangles[i]
			bb = l
			pose = "Unspecified"
			truncated = 0
			difficult = 0
			if len(l) > 5:

				truncated = l[4]
				difficult = l[5]


			text += """
\t<object>
\t\t<name>""" + saveTypes[i] + """</name>
\t\t<pose>""" + pose + """</pose>
\t\t<truncated>""" + str(truncated) + """</truncated>
\t\t<difficult>"""+ str(difficult) + """</difficult>
\t\t<bndbox>
\t\t\t<xmin>""" + str(min(bb[0], bb[2])) + """</xmin>
\t\t\t<xmax>""" + str(max(bb[0], bb[2])) + """</xmax>
\t\t\t<ymin>""" + str(min(bb[1], bb[3])) + """</ymin>
\t\t\t<ymax>""" + str(max(bb[1], bb[3])) + """</ymax>
\t\t</bndbox>"""

					
			text += "\n\t</object>"
			i+=1

		text += "\n</annotation>"


		fileName = fileName[0:-4] + ".xml"
		# print(text)
		if True:
			with open( folder + "/" + fileName, 'w') as fileHandle:
				fileHandle.write(text)
				fileHandle.close()


	def fixVideo(self, fileName):
		print("fixing the video. This might take a while, but you only have to do it once")
		oldFileName = fileName[0:-4] + "_corrupted.avi_bad"
		os.rename(fileName, oldFileName)
		cap = cv2.VideoCapture(oldFileName)

		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		   
		size = (frame_width, frame_height)

		frameCount = 0
		saveMethod = 'MJPG'
		colorWriter = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc(*saveMethod), 30, size)
		ret = cap.isOpened()
		while ret:
			
			ret, imgOG = cap.read()
			if ret:
				img = imgOG.copy()
				frameCount += 1
				if frameCount%10 == 0:
					cv2.imshow("img", img)
					colorWriter.write(img)
					k = cv2.waitKey(1)
					if k == 27:
						break

		cap.release()
		colorWriter.release()

		print("done")


if __name__ == "__main__":
	if modelName != "":
		from tflite_support.task import core
		from tflite_support.task import processor
		from tflite_support.task import vision
	annot = Annotator(videosPath, saveName, modelName)
	annot.goThroughDir()
