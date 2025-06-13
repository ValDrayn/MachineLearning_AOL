import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection import MeanAveragePrecision

# ========== DATASET DEFINITION ==========
class ParkingDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Returning None.")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path) as f:
                for line in f.readlines():
                    try:
                        cls, x_center, y_center, w, h = map(float, line.strip().split())
                        boxes.append([x_center, y_center, w, h])
                        labels.append(int(cls))
                    except ValueError:
                        print(f"Warning: Skipping malformed line in {label_path}")
                        continue

        target_for_transform = {'image': image, 'bboxes': boxes, 'labels': labels}

        if self.transform:
            augmented = self.transform(**target_for_transform)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']

        h, w = image.shape[1], image.shape[2]
        final_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            xmin = (x_center - width / 2) * w
            ymin = (y_center - height / 2) * h
            xmax = (x_center + width / 2) * w
            ymax = (y_center + height / 2) * h
            if xmax > xmin and ymax > ymin:
                final_boxes.append([xmin, ymin, xmax, ymax])

        if not final_boxes:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            final_boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
            
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if len(labels) != len(final_boxes):
             labels = labels[:len(final_boxes)]

        final_target = {"boxes": final_boxes, "labels": labels}

        return image, final_target

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return tuple(zip(*batch))

# ========== EVALUATION FUNCTION ==========
def get_validation_metrics(model, dataloader, device):
    model.eval()
    
    iou_thresholds_for_map = [0.45, 0.50, 0.75]
    metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=iou_thresholds_for_map).to(device)
    
    val_loss_total = 0.0
    val_loss_classifier = 0.0
    val_loss_box_reg = 0.0
    val_loss_objectness = 0.0
    val_loss_rpn_box_reg = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None or batch[0] is None: continue
            images, targets = batch

            images = [img.to(device) for img in images]
            targets_for_map = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            metric.update(predictions, targets_for_map)
            
            model.train()
            valid_targets_for_loss = [t for t in targets_for_map if t['boxes'].shape[0] > 0]
            if valid_targets_for_loss:
                valid_images_for_loss = [images[i] for i, t in enumerate(targets_for_map) if t['boxes'].shape[0] > 0]
                loss_dict_calc = model(valid_images_for_loss, valid_targets_for_loss)
                
                val_loss_total += sum(loss.item() for loss in loss_dict_calc.values())
                val_loss_classifier += loss_dict_calc.get('loss_classifier', torch.tensor(0.0)).item()
                val_loss_box_reg += loss_dict_calc.get('loss_box_reg', torch.tensor(0.0)).item()
                val_loss_objectness += loss_dict_calc.get('loss_objectness_rpn', torch.tensor(0.0)).item()
                val_loss_rpn_box_reg += loss_dict_calc.get('loss_rpn_box_reg', torch.tensor(0.0)).item()

            model.eval() 

    mAP_results = metric.compute()
    num_batches = len(dataloader)
    
    results = {
        "val/loss": val_loss_total / num_batches if num_batches > 0 else 0.0,
        "val/cls_loss": val_loss_classifier / num_batches if num_batches > 0 else 0.0,
        "val/box_loss": val_loss_box_reg / num_batches if num_batches > 0 else 0.0,
        "val/rpn_objectness_loss": val_loss_objectness / num_batches if num_batches > 0 else 0.0,
        "val/rpn_box_loss": val_loss_rpn_box_reg / num_batches if num_batches > 0 else 0.0,
        "metrics/precision": mAP_results.get("map", torch.tensor(-1.0)).item(),
        "metrics/mAP@0.45": mAP_results.get("map_iou", torch.tensor([-1.0]*len(iou_thresholds_for_map)))[0].item(),
        "metrics/mAP@0.50": mAP_results.get("map_50", torch.tensor(-1.0)).item(),
        "metrics/mAP@0.75": mAP_results.get("map_75", torch.tensor(-1.0)).item(),
        "metrics/recall": mAP_results.get("mar_100", torch.tensor(-1.0)).item(), 
    }
    return results

# ========== MAIN ==========
def main():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    train_dataset = ParkingDataset("dataset_1500_YOLO/train/images", "dataset_1500_YOLO/train/labels", transform=train_transform)
    val_dataset = ParkingDataset("dataset_1500_YOLO/val/images", "dataset_1500_YOLO/val/labels", transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", nms_thresh=0.45)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    resume_from_checkpoint = None 
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        model.load_state_dict(torch.load(resume_from_checkpoint))
    else:
        print("Starting training from scratch.")

    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 10
    log_file = 'training_log.csv'
    
    csv_headers = [
        "epoch", "time", 
        "train/loss", "train/cls_loss", "train/box_loss", "train/rpn_objectness_loss", "train/rpn_box_loss",
        "val/loss", "val/cls_loss", "val/box_loss", "val/rpn_objectness_loss", "val/rpn_box_loss",
        "metrics/precision", "metrics/mAP@0.45", "metrics/mAP@0.50", "metrics/mAP@0.75",
        "metrics/recall", "lr"
    ]
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        train_loss_total = 0.0
        train_loss_classifier = 0.0
        train_loss_box_reg = 0.0
        train_loss_objectness = 0.0
        train_loss_rpn_box_reg = 0.0
        
        num_valid_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            if batch is None or batch[0] is None: continue
            images, targets = batch

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            targets = [t for t in targets if t["boxes"].shape[0] > 0]
            if not targets:
                continue
            
            num_valid_batches += 1

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if torch.isnan(losses):
                print(f"\nWARNING: Detected NaN loss at epoch {epoch+1}. Skipping this batch.")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += losses.item()
            train_loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            train_loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            train_loss_objectness += loss_dict.get('loss_objectness_rpn', torch.tensor(0.0)).item()
            train_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()

            progress_bar.set_postfix(loss=losses.item())
            
        lr_scheduler.step()

        if num_valid_batches > 0:
            avg_train_losses = {
                "train/loss": train_loss_total / num_valid_batches,
                "train/cls_loss": train_loss_classifier / num_valid_batches,
                "train/box_loss": train_loss_box_reg / num_valid_batches,
                "train/objectness_loss": train_loss_objectness / num_valid_batches,
                "train/rpn_box_loss": train_loss_rpn_box_reg / num_valid_batches,
            }
        else:
            avg_train_losses = {k: 0.0 for k in ["train/loss", "train/cls_loss", "train/box_loss", "train/objectness_loss", "train/rpn_box_loss"]}


        val_results = get_validation_metrics(model, val_loader, device)
        
        checkpoint_path = f"result/models/parking_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                epoch + 1,
                round(time.time() - epoch_start_time, 2),
                round(avg_train_losses.get("train/loss", -1.0), 5),
                round(avg_train_losses.get("train/cls_loss", -1.0), 5),
                round(avg_train_losses.get("train/box_loss", -1.0), 5),
                round(avg_train_losses.get("train/objectness_loss", -1.0), 5),
                round(avg_train_losses.get("train/rpn_box_loss", -1.0), 5),

                round(val_results.get("val/loss", -1.0), 5),
                round(val_results.get("val/cls_loss", -1.0), 5),
                round(val_results.get("val/box_loss", -1.0), 5),
                round(val_results.get("val/rpn_objectness_loss", -1.0), 5),
                round(val_results.get("val/rpn_box_loss", -1.0), 5),

                round(val_results.get("metrics/precision", -1.0), 5),
                round(val_results.get("metrics/mAP@0.45", -1.0), 5),
                round(val_results.get("metrics/mAP@0.50", -1.0), 5),
                round(val_results.get("metrics/mAP@0.75", -1.0), 5),
                round(val_results.get("metrics/recall", -1.0), 5),
                optimizer.param_groups[0]['lr']
            ]
            writer.writerow(row)
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} Summary ---")
        print(f"Train Loss: {avg_train_losses.get('train/loss', -1):.4f} | Val mAP@0.50: {val_results.get('metrics/mAP@0.50', -1):.4f} | Val Recall: {val_results.get('metrics/recall', -1):.4f}")
        print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining finished.")

def visualize_prediction(model, dataset, device, num_images=10):
    model.eval()
    CONF_THRESHOLD = 0.25

    with open('predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score'])

        for i in range(num_images):
            image, _ = dataset[i]
            image_tensor = image.to(device).unsqueeze(0)
            with torch.no_grad():
                prediction = model(image_tensor)[0]

            scores = prediction['scores']
            
            keep = scores > CONF_THRESHOLD
            
            boxes = prediction['boxes'][keep].cpu().numpy()
            labels = prediction['labels'][keep].cpu().numpy()
            final_scores = scores[keep].cpu().numpy()

            img_for_vis = image.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_for_vis = std * img_for_vis + mean
            img_for_vis = np.clip(img_for_vis, 0, 1)

            plt.figure(figsize=(12, 8))
            plt.imshow(img_for_vis)
            

            for box, label, score in zip(boxes, labels, final_scores):
                # Tulis ke CSV
                writer.writerow([f'prediction_{i}.png'] + list(box) + [label, score])
                

                xmin, ymin, xmax, ymax = box
                color = "green" if label == 0 else "red"
                plt.gca().add_patch(plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    edgecolor=color, facecolor='none', linewidth=2
                ))
                plt.text(xmin, ymin - 5, f"Label: {label} ({score:.2f})", color=color, fontsize=10, backgroundcolor='white')

            plt.title(f"Prediction {i + 1} (Confidence > {CONF_THRESHOLD})")
            plt.axis('off')
            plt.savefig(f"prediction_{i}.png")
            plt.close()

if __name__ == "__main__":
    main()
