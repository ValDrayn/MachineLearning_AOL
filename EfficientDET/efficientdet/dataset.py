import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        # Ensure the annotation file exists before proceeding
        annotations_path = os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotation file not found: {annotations_path}")
            
        self.coco = COCO(annotations_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)

        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        img_name = image_info['file_name']

        sample = {'img': img, 'annot': annot, 'img_name': img_name} 

        if self.transform:
            sample = self.transform(sample)
            sample['img_name'] = img_name

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]

        # Construct the full path to the image
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])

        # Add this line to ensure path separators are consistent for the OS
        path = os.path.normpath(path) # <--- ENSURE THIS LINE IS PRESENT

        img = cv2.imread(path)

        if img is None:
            print(f"Warning: Could not load image from {path}. It might be missing or corrupted.")
            raise FileNotFoundError(f"Image not found or corrupted at: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

# Before calloter
# def collater(data):
#     imgs = [s['img'] for s in data]
#     annots = [s['annot'] for s in data]
#     scales = [s['scale'] for s in data]

#     imgs = torch.from_numpy(np.stack(imgs, axis=0))

#     max_num_annots = max(annot.shape[0] for annot in annots)

#     if max_num_annots > 0:

#         annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

#         for idx, annot in enumerate(annots):
#             if annot.shape[0] > 0:
#                 annot_padded[idx, :annot.shape[0], :] = annot
#     else:
#         annot_padded = torch.ones((len(annots), 1, 5)) * -1

#     imgs = imgs.permute(0, 3, 1, 2)

#     return {'img': imgs, 'annot': annot_padded, 'scale': scales}

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]  # <--- tambahkan ini

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {
        'img': imgs,
        'annot': annot_padded,
        'scale': scales,
        'img_name': img_names  # <--- tambahkan ini ke return
    }


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': sample.get('img_name')}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}