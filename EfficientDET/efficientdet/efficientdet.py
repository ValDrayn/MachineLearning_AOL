# Di dalam file utils/postprocess.py

import torch
import torch.nn as nn
import torchvision

from .utils import BBoxTransform, ClipBoxes

class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()
        self.bbox_transform = BBoxTransform()
        self.clip_boxes = ClipBoxes()

    def forward(self, regression, classification, anchors, imgs, conf_threshold=0.25, iou_threshold=0.45):

        transformed_anchors = self.bbox_transform(anchors, regression)
        transformed_anchors = self.clip_boxes(transformed_anchors, imgs)

        batch_scores = []
        batch_labels = []
        batch_boxes = []

        for i in range(imgs.shape[0]):
            classification_i = classification[i, :, :]
            transformed_anchors_i = transformed_anchors[i, :, :]
            
            scores = classification_i.sigmoid()
            
            scores_over_thresh, labels_over_thresh = torch.max(scores, dim=1)

            keep = scores_over_thresh > conf_threshold
            scores_over_thresh = scores_over_thresh[keep]

            if scores_over_thresh.shape[0] == 0:

                batch_scores.append(torch.tensor([]).to(imgs.device))
                batch_labels.append(torch.tensor([]).to(imgs.device))
                batch_boxes.append(torch.tensor([]).to(imgs.device))
                continue

            labels_over_thresh = labels_over_thresh[keep]
            boxes_over_thresh = transformed_anchors_i[keep, :]

            nms_indices = torchvision.ops.nms(boxes_over_thresh, scores_over_thresh, iou_threshold)

            batch_scores.append(scores_over_thresh[nms_indices])
            batch_labels.append(labels_over_thresh[nms_indices])
            batch_boxes.append(boxes_over_thresh[nms_indices])
            
        return batch_scores, batch_labels, batch_boxes