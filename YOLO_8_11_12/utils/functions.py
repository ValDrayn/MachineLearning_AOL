import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import cv2
import os
import shutil
import argparse
from typing import List

from PIL import Image

# ROOT DIRECTORY
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLOV5_VERSIONS = [
    "yolov5n.pt",
]

YOLOV8_VERSIONS = [
    "yolov8n.pt",
]

YOLOV10_VERSIONS = [
    "yolov10n.pt",
]

YOLOV11_VERSIONS = [
    "yolo11n.pt",
]

YOLOV12_VERSIONS = [
    "yolo12n.pt",
]

def draw_bounding_boxes(image_path, annotation_path, output_path, **kwargs):

    image = cv2.imread(image_path)
    
    try:
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
    except FileNotFoundError:
        annotations = []

    filled_spots = 0
    empty_spots = 0
    
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        
        img_h, img_w, _ = image.shape
        left = int((x_center - width / 2) * img_w)
        top = int((y_center - height / 2) * img_h)
        right = int((x_center + width / 2) * img_w)
        bottom = int((y_center + height / 2) * img_h)

        color = (0, 0, 0)
        

        if int(class_id) == 0: # 0: empty
            color = (0, 255, 0)
            empty_spots += 1
        elif int(class_id) == 1: # 1: filled
            color = (0, 0, 255)
            filled_spots += 1
            
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    total_spots = empty_spots + filled_spots
    def draw_text_with_background(img, text, position, font, scale, text_color, bg_color, alpha=0.6, thickness=2, padding=10):
        overlay = img.copy()
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size

        x, y = position
        cv2.rectangle(
            overlay,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding),
            bg_color,
            -1
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


        cv2.putText(img, text, (x, y), font, scale, text_color, thickness)


    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    text_color = (0, 0, 0)
    bg_color = (200, 200, 200)  
    thickness = 2
    alpha = 0.6


    draw_text_with_background(image, f'Filled Spots: {filled_spots}', (30, 50), font, scale, text_color, bg_color, alpha, thickness)
    draw_text_with_background(image, f'Empty Spots: {empty_spots}', (30, 100), font, scale, text_color, bg_color, alpha, thickness)
    draw_text_with_background(image, f'Total Spots: {total_spots}', (30, 150), font, scale, text_color, bg_color, alpha, thickness)
    

    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    results = pd.DataFrame({
        'Image File': [os.path.basename(image_path)],
        'Empty Spots': [empty_spots],
        'Filled Spots': [filled_spots],
        'Total Spots Detected': [total_spots]
    })
        
    return results


def process_labels(data_path, new_labels_folder):
    print(f"Location: {new_labels_folder}")
    
    if not os.path.exists(new_labels_folder):
        raise FileNotFoundError(f"Not Found: {new_labels_folder}")
        
    return new_labels_folder


def process_images(data_path: str, output_folder: str, model: str = ''): 
    processed_images = 0

    images_folder = os.path.join(data_path, 'images/')

    if model != '':
        # If model is a path
        if "/" in model or "\\" in model:
            new_labels_folder = model
        else: # if model is a name
            new_labels_folder = os.path.join(ROOT, f'results/{model}/labels/')

        try: 
            print(f'Using labels from {new_labels_folder}')

            labels_folder = process_labels(data_path, new_labels_folder)
        except FileNotFoundError as e:
            print(e)
            print(
                f'No labels found for model {model}!\n' +
                'Make sure you wrote the correct model name. Otherwise, train the model first.')
            return pd.DataFrame()
    
    else:
        labels_folder = os.path.join(data_path, 'labels/')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_images_folder = os.path.join(output_folder, 'images/')
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    results_df = pd.DataFrame()

    print('Processing images...')
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(labels_folder, annotation_file)
        
        if os.path.exists(annotation_path):
            results = draw_bounding_boxes(image_path, annotation_path, output_images_folder)
            processed_images += 1
            results_df = pd.concat([results_df, results], ignore_index=True)
        else:
            print(f"No detections for {image_file}, skipping visualization.")

    print(f'Processed {processed_images} images ✅')
    results_df.to_csv(output_folder + 'output.csv', index=False)

    return results_df

def is_custom_model(model: str, yoloversion: str):
    if not model.endswith(".pt"):
        model = model + ".pt"

    YOLO_VERSIONS = {
        "8": YOLOV8_VERSIONS,
        "11": YOLOV11_VERSIONS,
        "12": YOLOV12_VERSIONS
    }

    versions = YOLO_VERSIONS.get(yoloversion, YOLOV8_VERSIONS)

    if not model in versions:
        if model.__contains__("/") or model.__contains__("\\"):
            model_path = model 
            model = model.split("/")[-1]
            model = model.split("\\")[-1]
        else: 
            model_path = os.path.join(ROOT, f"models/{model}")

    else:
        model_path = model

    return model, model_path

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reporoot', type=str, default=ROOT, help='path to repo root')
    opt = parser.parse_args()
    return opt

def mean_df(df: pd.DataFrame):
    columns_to_mean = ['Precision', 'Recall', 'mAP0-50', 'mAP50-95']

    means = []
    for i in range(0, len(df), 2):
        avg = df.iloc[i:i+2][columns_to_mean].mean()
        means.append(avg)

    new_data = {
        'Model': ['YOLOv5n', 'YOLOv5s', 'YOLOv8n', 'YOLOv8s'],
        'Model Size (MB)': [df.iloc[0]['Model Size (MB)'], df.iloc[2]['Model Size (MB)'], df.iloc[4]['Model Size (MB)'], df.iloc[6]['Model Size (MB)']],
        'Parameters': [df.iloc[0]['Parameters'], df.iloc[2]['Parameters'], df.iloc[4]['Parameters'], df.iloc[6]['Parameters']],
        'Precision': [mean['Precision'] for mean in means],
        'Recall': [mean['Recall'] for mean in means],
        'mAP0-50': [mean['mAP0-50'] for mean in means],
        'mAP50-95': [mean['mAP50-95'] for mean in means]
    }

    return pd.DataFrame(new_data)