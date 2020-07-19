# Write Python3 code here 
import os 
import cv2 
import numpy as np 
import tensorflow as tf 

import sys 
import pytesseract
# This is needed since the notebook is stored in the object_detection folder. 
sys.path.append("..") 

def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result


# Import utilites 
from utils import label_map_util 
from utils import visualization_utils as vis_util 

# Name of the directory containing the object detection module we're using 
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28' # The path to the directory where frozen_inference_graph is stored. 
IMAGE_NAME = 'image1.jpg' # The path to the image in which the object has to be detected. 

# Grab path to current working directory 
CWD_PATH = os.getcwd() 

# Path to frozen detection graph .pb file, which contains the model that is used 
# for object detection. 
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb') 

# Path to label map file 
PATH_TO_LABELS = os.path.join('/home/capeie/marathon/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

# Path to image 
PATH_TO_IMAGE = os.path.join(CWD_PATH+"/test_images", IMAGE_NAME) 

# Number of classes the object detector can identify 
NUM_CLASSES = 90

# Load the label map. 
# Label maps map indices to category names, so that when our convolution 
# network predicts `5`, we know that this corresponds to `king`. 
# Here we use internal utility functions, but anything that returns a 
# dictionary mapping integers to appropriate string labels would be fine 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS) 
categories = label_map_util.convert_label_map_to_categories( 
		label_map, max_num_classes = NUM_CLASSES, use_display_name = True) 
category_index = label_map_util.create_category_index(categories) 

# Load the Tensorflow model into memory. 
detection_graph = tf.Graph() 
with detection_graph.as_default(): 
	od_graph_def = tf.GraphDef() 
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
		serialized_graph = fid.read() 
		od_graph_def.ParseFromString(serialized_graph) 
		tf.import_graph_def(od_graph_def, name ='') 

	sess = tf.Session(graph = detection_graph) 

# Define input and output tensors (i.e. data) for the object detection classifier 

# Input tensor is the image 
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') 

# Output tensors are the detection boxes, scores, and classes 
# Each box represents a part of the image where a particular object was detected 
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 

# Each score represents level of confidence for each of the objects. 
# The score is shown on the result image, together with the class label. 
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') 
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 

# Number of objects detected 
num_detections = detection_graph.get_tensor_by_name('num_detections:0') 

# Load image using OpenCV and 
# expand image dimensions to have shape: [1, None, None, 3] 
# i.e. a single-column array, where each item in the column has the pixel RGB value 
image = cv2.imread(PATH_TO_IMAGE) 
image_expanded = np.expand_dims(image, axis = 0) 

# Perform the actual detection by running the model with the image as input 
(boxes, scores, classes, num) = sess.run( 
	[detection_boxes, detection_scores, detection_classes, num_detections], 
	feed_dict ={image_tensor: image_expanded}) 



# Draw the results of the detection (aka 'visualize the results') 

vis_util.visualize_boxes_and_labels_on_image_array( 
	image, 
	np.squeeze(boxes), 
	np.squeeze(classes).astype(np.int32), 
	np.squeeze(scores), 
	category_index, 
	use_normalized_coordinates = True, 
	line_thickness = 8, 
	min_score_thresh = 0.60) 

# All the results have been drawn on the image. Now display the image. 
cv2.imshow('Object detector', image) 
image = cv2.resize(image, (512,512), interpolation = cv2.INTER_AREA)
shapex, shapey, _ = image.shape
print(image.shape)
#heavy testing
for i,b in enumerate(boxes[0]):
	x1 = int(remap(boxes[0][i][1],0,1.0,0,512))
	x2 = int(remap(boxes[0][i][3],0,1.0,0,512))
	y1 = int(remap(boxes[0][i][0],0,1.0,0,512))
	y2 = int(remap(boxes[0][i][2],0,1.0,0,512))
	if x1!=0 and x2!=0:
		roi=image[y1:y2,x1:x2]
		cv2.imshow('roi',image[y1:y2,x1:x2])
		cv2.imwrite("roi-" + str(i) + ".jpg", roi)
		
    

# Press any key to close the image 
cv2.waitKey(0) 

# Clean up 
cv2.destroyAllWindows() 


# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import imutils
import argparse
import cv2
import shutil


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # etract the scores (probabilites), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x*4.0, y*4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)- coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin*xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos*xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=800,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=1184,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread('roi-1.jpg')
orig = image.copy()
(origH, origW) = image.shape[:2]
print (orig.shape)
# set the new width and height and then determine the ratio in change for both
(newW, newH) = (args["width"],args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested in -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('models/frozen_east_text_detection.pb')

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []

#loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    #scale the bounding boxes coordinates based on the respective ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bb - here we are computing deltas
    # in both the x and y directions
    dX = int((endX - startX) * args["padding"])
    dY = int((endY - startY) * args["padding"])

    # apply padding to each side of the bb,respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
    roi = orig[startY:endY, startX:endX]
    print(startY, endY, startX, endX)
    # Applying Tesseract v4 to OCR flags:
    # 1. langauge
    # 2. OEM flag: 4 - use LSTM neural net model for OCR
    # 3. OEM value: 7 - treating ROI as singleline of text
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)

    # add the bounding box coordinates and OCR'd text to the list of results
    results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

# loop over the results
tex = []
for ((startX, startY, endX, endY), text) in results:
    #displaythe text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text));tex.append(text)
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding
    # the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY), (0,0,255), 2)
    cv2.putText(output, text, (startX, startY-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    #show the output image
    cv2.imshow("Text Detection", imutils.resize(output, height=800))
    cv2.waitKey(0)


parent = '/home/capeie/marathon/models/research/MaraPriv'
directory = tex[0]
mode = 0o666
fin = os.path.join(parent,directory)
os.makedirs(fin,mode)
source = 'test_images/curefit.jpg'

shutil.move(source, fin)
cv2.destroyAllWindows()