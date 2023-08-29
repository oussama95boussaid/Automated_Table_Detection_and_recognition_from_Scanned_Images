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

# Finding the cells & extracting the text using OCR from the table

**Removing The Lines**

This stage is mainly to delete table Lines. This will help us get a clear picture of the OCR process. In the end, only the text in the table cells remains in the image.
