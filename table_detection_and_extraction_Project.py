# Table Detection and Extraction From Image Project
# Step 1 : Detecting the table from the image 


import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
from ultralyticsplus import YOLO, render_result
from PIL import Image #pillow
import cv2
import subprocess

image = 'test4.jpg'
img = Image.open(image)

# load model
model = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# perform inference
results = model.predict(img)

img = np.array(Image.open(image))
tables_vesul=[]
tables=[]

for i in range(len(results[0].boxes.data.numpy())):
  x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[i]) # (96, 586, 947, 1286)
  x2+=10
  y2+=10
  x1-=10
  y1-=10

  #cropping
  cropped_image = img[y1:y2, x1:x2]
  print(cropped_image.shape)
  tables.append(cropped_image)
  image_tab = Image.fromarray(cropped_image)
  tables_vesul.append(image_tab)
  

def add_10_percent_padding(img):
    image_height = img.shape[0]
    padding = int(image_height * 0.1)
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded_img

def img_Preprocessing(img):
  # Grey-scaling
  grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # converting it to binary image by Thresholding
  thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]
  # Inverting
  inverted_image = cv2.bitwise_not(thresholded_image)
  return inverted_image

def img_Preprocessing_ocr(img):
  # Thresholding
  thresholded_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
  # Inverting
  # inverted_image = cv2.bitwise_not(thresholded_image)
  return thresholded_image

# Combining Vertical And Horizontal Lines
def combine_eroded_images(img):
  # Eroding Vertical Lines
  hor = np.array([[1,1,1,1,1,1]])
  ver_erode_img = cv2.erode(img, hor, iterations=5)
  ver_dilate_img = cv2.dilate(ver_erode_img, hor, iterations=10)
  # Eroding Horizontal Lines
  ver = np.array([[1],[1],[1],[1],[1],[1],[1]])
  hor_erode_img = cv2.erode(img, ver, iterations=5)
  hor_dilate_img = cv2.dilate(hor_erode_img, ver, iterations=10)
  # Combining
  combined_image = cv2.add(ver_dilate_img, hor_dilate_img)
  # dilate combined_image to make lines thicker
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)
  return combined_image_dilated

# Removing The Lines
def subtract_combined_and_dilated_image_from_original_image(processed_img,combined_image_dilated):
    img_without_lines = cv2.subtract(processed_img,combined_image_dilated)
    # remove noise with erode and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_without_lines_noise_removed = cv2.erode(img_without_lines, kernel)
    image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel)
    return image_without_lines_noise_removed



