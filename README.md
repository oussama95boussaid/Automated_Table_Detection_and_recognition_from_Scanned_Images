# **Automated Table Detection and Recognition from Scanned Images**

**Project Overview**

The **Automated Table Detection and Recognition from Scanned Images project** is a sophisticated algorithm designed to accurately identify tables within scanned images. Its core objective is to overcome challenges related to diverse layouts, fonts, and varying image quality levels. This algorithm not only excels at locating tables but also seamlessly extracts valuable data from them.

# Key Features

- **Precise Table Identification** : Our algorithm can precisely locate tables within scanned images, even in cases with complex layouts and diverse fonts.

- **Robust Image Quality Handling** : It's capable of handling varying image quality levels, ensuring reliable performance across different scanned documents.

- **Data Extraction** : Beyond table detection, this algorithm excels at extracting data from these tables, making it a comprehensive tool for data analysis.

# Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

# Prerequisites

To run this project, you'll need:

- *tesseract-ocr-fra*
- *ultralyticsplus (version 0.0.23)*
- *ultralytics (version 8.0.21)*
- *opencv2*
- *pandas*
- *csv*
- *subprocess*
- *PIL*

# Installation

    git clone https://github.com/oussama95boussaid/Automated_Table_Detection_and_recognition_from_Scanned_Images_PDFs.git
    cd Automated_Table_Detection_and_recognition_from_Scanned_Images_PDFs

# Install the required libraries for linux:

    !sudo apt install tesseract-ocr-fra
    !pip install pytesseract transformers ultralyticsplus==0.0.23 ultralytics==8.0.21

# Run the algorithm:

    python table_detection_and_extraction_Project.py

# Project Stepts :

-  step 1 : Detecting the table 
-  step 2 : Extract table from the image
-  step 3 : Finding the cells & extracting the text using OCR
-  step 4 : Generating The CSV file

# Detecting the table

In the process of our project, we utilized the YOLOv8 model with specific parameter configurations to tackle the critical task of table detection. This step was pivotal in our workflow, as accurate table detection is a fundamental component of various computer vision applications.

To optimize our model's performance, we fine-tuned the following parameters:

- Confidence Threshold (conf): We set this to 0.25, which determined the minimum confidence level a detected object had to meet to be considered valid. Lower values can result in more detections but might also increase false positives.

- Intersection over Union Threshold (iou): We configured this to 0.45, which controlled the overlap allowed between different bounding boxes during Non-Maximum Suppression (NMS). A higher value can lead to more consolidated bounding boxes.

- Class-Agnostic NMS (agnostic_nms): We kept this parameter set to False, meaning that NMS was class-specific. In other words, objects of different classes were treated separately during NMS.

- Maximum Detections per Image (max_det): We limited the number of detections per image to 1000. This helped manage the computational load and memory usage, especially in situations where numerous objects were present in an image.

These parameter settings were meticulously chosen to strike a balance between precision and recall, ensuring that our table detection model performed effectively and efficiently.

<img src = "img_Preprocessing/combined_org_dete_img.png" >

After detecting the table that's the result of cropping it from the image based on boxes

<img src = "img_Preprocessing/extracted_img.png" >

# Finding the cells & extracting the text using OCR from the table

**Data Preprocessing**

This stage **Removing The Lines** is mainly to delete table Lines. 

This will help us get a clear picture of the OCR process. In the end, only the text in the table cells remains in the image.

1. **add 10 percent padding** :

   This will be needed in the next stage when we remove the lines Without this, the lines do not get removed fully
 
2. **Grey-scaling & Thresholding &  Inverting** :

   This stage takes the full color image plus padding from the last stage and converts it to an inverted binary image

 Result of step 1 & 2 

 <img src = "img_Preprocessing/processed_img.png" >

3. **Eroding Vertical Lines** :

   To understand how vertical lines and all text erode, you need to understand the concepts of "*erosion*" and "*dilation*" properly.*“kernel”* is in the context of erosion and dilation. Basically, it’s a shape that is taken over the images and used to transform the 
   underlying image by removing or adding pixels to the original image.for more information <a href = "https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html">Extract horizontal and vertical lines by using morphological operations </a>
   <img src = "img_Preprocessing/ver_dilate_img.png" >

4. **Eroding Horizontal Lines** :

   Similar process to erode away the horizontal lines
   <img src = "img_Preprocessing/hor_dilate_img.png" >

5. **Combining Vertical And Horizontal Lines** :

   Combine the horizontal and vertical lines using a simple *add* operation. It just adds the white pixels in both image, I used *dilate* once again to “thicken” these lines, befor that i used **getStructuringElement** in order to create a nice simple rectangular 
   kernel. The simple kernel will go over the image and thicken things up
   <img src = "img_Preprocessing/combined_img.png" >




   

