import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from tqdm import tqdm

def load_model(checkpoint_path, num_classes=2, nms_thresh=0.45):
    model = fasterrcnn_resnet50_fpn(weights=None, nms_thresh=nms_thresh)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    
    return model

def predict_on_image(model, image_path, device, confidence_threshold=0.25):
    model.to(device)
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    image_pil_to_draw = Image.fromarray(image_rgb)

    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    draw = ImageDraw.Draw(image_pil_to_draw)
    
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']

    for i, box in enumerate(boxes):
        score = scores[i].item()
        if score > confidence_threshold:
            box_coords = [coord.item() for coord in box]
            label = int(labels[i].item())
            
            if label == 0:
                label_text = "Empty"
                color = "green"
            else:
                label_text = "Filled"
                color = "red"
            
            draw.rectangle(box_coords, outline=color, width=3)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            text = f"{label_text} ({score:.2f})"
            text_bbox = draw.textbbox((box_coords[0], box_coords[1] - 25), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((box_coords[0], box_coords[1] - 25), text, fill="white", font=font)
            
    return image_pil_to_draw

if __name__ == '__main__':
    CHECKPOINT_FILE = "parking_model_epoch_10.pth"
    TEST_IMAGE_FOLDER = "dataset_1500_YOLO/test/images" 
    OUTPUT_FOLDER = "result/images"                    
    CONFIDENCE = 0.25
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from {CHECKPOINT_FILE}...")
    model = load_model(CHECKPOINT_FILE, num_classes=2, nms_thresh=0.45)
    
    image_files = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to predict in {TEST_IMAGE_FOLDER}")

    for image_name in tqdm(image_files, desc="Predicting images"):
        image_path = os.path.join(TEST_IMAGE_FOLDER, image_name)
        output_path = os.path.join(OUTPUT_FOLDER, f"pred_{image_name}")

        try:
            predicted_image = predict_on_image(model, image_path, device, confidence_threshold=CONFIDENCE)
            
            if predicted_image:
                predicted_image.save(output_path)
        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")
            
    print(f"\nAll predictions saved to {OUTPUT_FOLDER}")
