import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import time
import csv
from torchmetrics.detection import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.optim.lr_scheduler import StepLR
import cv2
from tqdm import tqdm

# ========== DATASET DEFINITION ==========
class ParkingDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        # Baca gambar menggunakan cv2 karena Albumentations bekerja dengan array NumPy
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path) as f:
                for line in f.readlines():
                    # Bounding box harus dalam format [xmin, ymin, xmax, ymax]
                    cls, x_center, y_center, w, h = map(float, line.strip().split())
                    # Koordinat sudah dalam format YOLO, jadi kita simpan apa adanya
                    # Albumentations akan mengonversinya
                    boxes.append([x_center, y_center, w, h])
                    labels.append(int(cls))

        # Buat dictionary target untuk augmentasi
        target = {'image': image, 'bboxes': boxes, 'labels': labels}

        # Terapkan augmentasi
        if self.transform:
            augmented = self.transform(**target)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']

        # Konversi bounding box ke format [xmin, ymin, xmax, ymax] yang dibutuhkan FasterRCNN
        h, w, _ = image.shape
        final_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            xmin = (x_center - width / 2) * w
            ymin = (y_center - height / 2) * h
            xmax = (x_center + width / 2) * w
            ymax = (y_center + height / 2) * h
            final_boxes.append([xmin, ymin, xmax, ymax])

        # Konversi ke tensor
        final_boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        final_target = {"boxes": final_boxes, "labels": labels}

        return image, final_target

def collate_fn(batch):
    return tuple(zip(*batch))

# ========== EVALUATION FUNCTION ==========
def get_validation_metrics(model, dataloader, device):
    """
    Menghitung loss validasi dan mAP dengan benar dengan menangani mode model.
    """
    # Mulai dalam mode eval untuk konsistensi
    model.eval()
    
    iou_thresholds_for_map = [0.45, 0.50, 0.75]
    metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=iou_thresholds_for_map).to(device)
    
    val_loss_total = 0.0
    val_loss_classifier = 0.0
    val_loss_box_reg = 0.0
    val_loss_objectness = 0.0
    val_loss_rpn_box_reg = 0.0
    
    # torch.no_grad() sangat penting: ini mencegah perhitungan gradien
    # dan memastikan statistik batch norm tidak diperbarui, bahkan dalam mode train.
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets_for_map = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # --- Langkah 1: Dapatkan prediksi untuk mAP (model dalam mode eval) ---
            # Kita panggil model tanpa target untuk mendapatkan HANYA prediksi
            predictions = model(images)
            metric.update(predictions, targets_for_map)
            
            # --- Langkah 2: Dapatkan dictionary loss (alih sementara ke mode train) ---
            # Ini adalah cara standar untuk mendapatkan loss validasi pada model deteksi torchvision.
            model.train()
            loss_dict_calc = model(images, targets_for_map)
            model.eval() # Segera kembalikan ke mode eval
            
            # Sekarang, loss_dict_calc dijamin berupa dictionary berisi loss
            val_loss_total += sum(loss for loss in loss_dict_calc.values()).item()
            val_loss_classifier += loss_dict_calc.get('loss_classifier', torch.tensor(0)).item()
            val_loss_box_reg += loss_dict_calc.get('loss_box_reg', torch.tensor(0)).item()
            # Gunakan 'loss_objectness_rpn' yang merupakan kunci yang benar dari model
            val_loss_objectness += loss_dict_calc.get('loss_objectness_rpn', torch.tensor(0)).item() 
            val_loss_rpn_box_reg += loss_dict_calc.get('loss_rpn_box_reg', torch.tensor(0)).item()

    mAP_results = metric.compute()
    num_batches = len(dataloader)
    
    # Mengambil mAP@0.45 dengan cara yang lebih andal
    # 'map_iou' adalah tensor yang berisi mAP untuk setiap IoU yang kita tentukan
    map_at_45 = mAP_results["map_iou"][0].item()

    results = {
        "val/loss": val_loss_total / num_batches,
        "val/cls_loss": val_loss_classifier / num_batches,
        "val/box_loss": val_loss_box_reg / num_batches,
        "val/rpn_objectness_loss": val_loss_objectness / num_batches,
        "val/rpn_box_loss": val_loss_rpn_box_reg / num_batches,
        "metrics/precision": mAP_results["map"].item(),
        "metrics/mAP@0.45": map_at_45,
        "metrics/mAP@0.50": mAP_results["map_50"].item(),
        "metrics/mAP@0.75": mAP_results["map_75"].item(),
        "metrics/recall": mAP_results["mar_100"].item(), 
    }
    return results

# ========== MAIN ==========
def main():
    ### PERUBAHAN 1: Definisikan pipeline augmentasi ###
    # Pipeline untuk data training: menyertakan augmentasi acak
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        # Normalisasi menggunakan nilai standar ImageNet
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Konversi ke tensor PyTorch
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    # Pipeline untuk data validasi: hanya normalisasi dan konversi tensor
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


    ### PERUBAHAN 2: Gunakan transform yang baru saat membuat dataset ###
    # Hapus baris lama: transform = ToTensor()
    train_dataset = ParkingDataset("dataset_1500_YOLO/train/images", "dataset_1500_YOLO/train/labels", transform=train_transform)
    val_dataset = ParkingDataset("dataset_1500_YOLO/val/images", "dataset_1500_YOLO/val/labels", transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # --- (Bagian inisialisasi model dan pengecekan CUDA tetap sama) ---
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", nms_thresh=0.45)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
        try:
            print("\n--- Running Torchvision CUDA Test ---")
            boxes1 = torch.tensor([[0, 0, 20, 20], [10, 10, 30, 30]], dtype=torch.float32).cuda()
            boxes2 = torch.tensor([[5, 5, 25, 25]], dtype=torch.float32).cuda()
            iou_matrix = torchvision.ops.box_iou(boxes1, boxes2)
            print("Torchvision CUDA operations are working correctly.")
            print("IOU Matrix calculated on GPU:")
            print(iou_matrix)
        except Exception as e:
            print(f"\n--- Test Failed ---")
            print(f"An error occurred during the torchvision CUDA test: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- (Bagian resume training tetap sama) ---
    resume_from_checkpoint = None 
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        model.load_state_dict(torch.load(resume_from_checkpoint))
    else:
        print("Starting training from scratch.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    ### PERUBAHAN 3: Definisikan Learning Rate Scheduler ###
    # Scheduler ini akan mengurangi learning rate sebesar 10x (gamma=0.1) setiap 10 epoch (step_size=10)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 10 # Direkomendasikan menambah epoch saat menggunakan augmentasi
    log_file = 'training_log.csv'
    
    # --- (Bagian header CSV tetap sama) ---
    csv_headers = [
        "epoch", "time", 
        "train/loss", "train/cls_loss", "train/box_loss", "train/rpn_objectness_loss", "train/rpn_box_loss",
        "val/loss", "val/cls_loss", "val/box_loss", "val/rpn_objectness_loss", "val/rpn_box_loss",
        "metrics/precision", 
        "metrics/mAP45", "metrics/mAP50", "metrics/mAP75", # Sesuaikan dengan metrik baru
        "metrics/recall", "lr"
    ]
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # --- (Loop training per batch tetap sama) ---
        train_loss_total = 0.0
        train_loss_classifier = 0.0
        train_loss_box_reg = 0.0
        train_loss_objectness = 0.0
        train_loss_rpn_box_reg = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # for images, targets in train_loader:
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss_total += losses.item()
            train_loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0)).item()
            train_loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0)).item()
            train_loss_objectness += loss_dict.get('loss_objectness_rpn', torch.tensor(0)).item()
            train_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item()

            progress_bar.set_postfix(loss=losses.item())
            
        num_train_batches = len(train_loader)
        avg_train_losses = {
            "train/loss": train_loss_total / num_train_batches,
            "train/cls_loss": train_loss_classifier / num_train_batches,
            "train/box_loss": train_loss_box_reg / num_train_batches,
            "train/objectness_loss": train_loss_objectness / num_train_batches,
            "train/rpn_box_loss": train_loss_rpn_box_reg / num_train_batches,
        }

        ### PERUBAHAN 4: Panggil scheduler setelah epoch selesai ###
        lr_scheduler.step()

        val_results = get_validation_metrics(model, val_loader, device)
        
        checkpoint_path = f"parking_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # --- (Bagian logging ke CSV dan print ke konsol tetap sama) ---
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                epoch + 1,
                round(time.time() - epoch_start_time, 2),
                round(avg_train_losses["train/loss"], 5),
                round(avg_train_losses["train/cls_loss"], 5),
                round(avg_train_losses["train/box_loss"], 5),
                round(avg_train_losses["train/objectness_loss"], 5),
                round(avg_train_losses["train/rpn_box_loss"], 5),
                round(val_results["val/loss"], 5),
                round(val_results["val/cls_loss"], 5),
                round(val_results["val/box_loss"], 5),
                round(val_results["val/rpn_objectness_loss"], 5),
                round(val_results["val/rpn_box_loss"], 5),
                round(val_results["metrics/precision"], 5),
                round(val_results["metrics/mAP45"], 5),      # TAMBAHKAN INI
                round(val_results["metrics/mAP50"], 5),      # Pastikan ini ada di results
                round(val_results["metrics/mAP75"], 5),      # Pastikan ini ada di results
                round(val_results["metrics/recall"], 5),
                optimizer.param_groups[0]['lr']
            ]
            writer.writerow(row)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_losses['train/loss']:.4f} | "
              f"Val mAP: {val_results['metrics/precision']:.4f} | "
              f"Val Recall: {val_results['metrics/recall']:.4f} | "
              f"Saved to {checkpoint_path}")

    print("Training finished.")


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

            # 1. Ambil scores dari prediksi
            scores = prediction['scores']
            
            # 2. Buat filter/mask untuk prediksi dengan score di atas ambang batas
            keep = scores > CONF_THRESHOLD
            
            # 3. Terapkan filter pada boxes, labels, dan scores
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
            
            # Loop melalui hasil yang sudah difilter
            for box, label, score in zip(boxes, labels, final_scores):
                # Tulis ke CSV
                writer.writerow([f'prediction_{i}.png'] + list(box) + [label, score])
                
                # Gambar kotak dan label pada gambar
                xmin, ymin, xmax, ymax = box
                color = "green" if label == 0 else "red"
                plt.gca().add_patch(plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    edgecolor=color, facecolor='none', linewidth=2
                ))
                # Tambahkan label dengan score
                plt.text(xmin, ymin - 5, f"Label: {label} ({score:.2f})", color=color, fontsize=10, backgroundcolor='white')

            plt.title(f"Prediction {i + 1} (Confidence > {CONF_THRESHOLD})")
            plt.axis('off')
            plt.savefig(f"prediction_{i}.png")
            plt.close()

if __name__ == "__main__":
    main()