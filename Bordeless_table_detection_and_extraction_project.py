import numpy as np
import pandas as pd
import re
import cv2
import csv
from ultralyticsplus import YOLO, render_result
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image #pillow

anaylise = cv2.imread("anaylise1.jpg")  #RGB
anaylise1 = Image.fromarray(anaylise)

# load model
model = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# perform inference
results = model.predict(anaylise1)

# select RIO
# img = np.array(Image.open(anaylise))

def RIO_select(img):
    x1, y1, x2, y2, _, _ = tuple(int(item) for item in results[0].boxes.data.numpy()[0]) # (96, 586, 947, 1286)
    # x2+=10
    # y2-=310
    # x1-=10
    # y1+=170
    #cropping
    cropped_image = img[y1:y2, x1:x2]
    print(cropped_image.shape)
    return cropped_image

tables = RIO_select(anaylise)

def add_10_percent_padding(img):
    image_height = img.shape[0]
    padding = int(image_height * 0.1)
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded_img
def generate_csv_file(input_table):
    with open("./outputs/final_result.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in input_table:
            csv_writer.writerow(row)

# ocr for text extraction
ocr = PaddleOCR(lang='en')
# Image preprocessing
image_with_padding = add_10_percent_padding(tables)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erode_img = cv2.erode(image_with_padding,kernel)
processed_img = cv2.cvtColor(erode_img, cv2.COLOR_BGR2RGB)

result_ = ocr.ocr(processed_img)

# select text extracted
result_extracted = result_[0]
text_extracted = [row[1][0] for row in result_extracted]

EDTA_list=[]
stack=[]
table=[]

# Extract EDAT
for ele in text_extracted:
  EDTA = re.findall(r'^[a-zA-Z. é]+$',ele,flags=re.IGNORECASE)
  if (len(EDTA) != 0):
      EDTA_list.append(EDTA[0])

# Edit text extracted
cleane_table = text_extracted.copy()
for i,ele in enumerate(text_extracted):
    if ele == '/mm3' or ele == 'mm3' :
       if text_extracted[i-1] in EDTA_list:
          ele = text_extracted[i-2]+ele
          cleane_table.insert(i,ele)
          del cleane_table[i-2]
       else:
          ele = text_extracted[i-1]+ele
          cleane_table.insert(i,ele)
          del cleane_table[i-1]

cleaned_data = [i for n, i in enumerate(cleane_table) if i not in cleane_table[:n]]
cleaned_data.remove('/mm3')

# Creat table To generate CSV file
for ele in cleaned_data:
  ele = ele.replace('g7', 'g/')
  ele = ele.replace('f1', 'fl')
  if ele in EDTA_list[1:]:
    table.append(stack)
    stack=[]
  if ele == EDTA_list[-1]:
    table.append(stack)
  stack.append(ele.strip('.'))

# Edit table
# add header
header = ['EDTA','Resultats','V.reference','Anterieurs','Date']
table.insert(0,header)

#*****
table[13].insert(1,table[12][-1])
del table[12][-2:]

#****
table[-1].insert(1,table[-2][-1])
table[-2].remove(table[-2][-1])

# save data in CSV file
generate_csv_file(table)
