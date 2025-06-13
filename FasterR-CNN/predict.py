# predict.py

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor

def load_model(checkpoint_path, num_classes=2):
    """
    Loads a model from a .pth checkpoint file.
    """
    # Define the model architecture (must be the same as when trained)
    model = fasterrcnn_resnet50_fpn(weights=None) # Don't use pretrained weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load the trained weights
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model

def predict_on_image(model, image_path, device, confidence_threshold=0.5):
    """
    Runs inference on a single image and draws the predicted boxes.
    """
    model.to(device)
    model.eval() # Set the model to evaluation mode

    image = Image.open(image_path).convert("RGB")
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    # --- Draw the boxes on the image ---
    draw = ImageDraw.Draw(image)
    
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']

    for i, box in enumerate(boxes):
        score = scores[i].item()
        if score > confidence_threshold:
            box = [b.item() for b in box]
            label = int(labels[i].item())
            
            # Assuming class 0 is parking spot
            color = "green" if label == 0 else "red" 
            draw.rectangle(box, outline=color, width=3)
            
            # Optional: Draw label and score
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            text = f"Label: {label}, Score: {score:.2f}"
            draw.text((box[0], box[1] - 20), text, fill=color, font=font)
            
    return image


if __name__ == '__main__':
    # --- CONFIGURATION ---
    CHECKPOINT_FILE = "parking_model_epoch_20.pth"  # <-- The .pth file you want to use
    IMAGE_TO_PREDICT = "path/to/your/test_image.jpg" # <-- The image you want to test
    OUTPUT_IMAGE_FILE = "prediction_output.jpg"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the model from the checkpoint
    print(f"Loading model from {CHECKPOINT_FILE}...")
    model = load_model(CHECKPOINT_FILE)
    
    # 2. Run prediction
    print(f"Running prediction on {IMAGE_TO_PREDICT}...")
    predicted_image = predict_on_image(model, IMAGE_TO_PREDICT, device)
    
    # 3. Save or show the result
    predicted_image.save(OUTPUT_IMAGE_FILE)
    predicted_image.show()
    
    print(f"Prediction saved to {OUTPUT_IMAGE_FILE}")