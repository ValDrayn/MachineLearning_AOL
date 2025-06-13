import torch
import itertools
import math

class Anchors:
    def __init__(self, pyramid_levels=[3, 4, 5, 6, 7],
                 scales=None, ratios=None,
                 image_size=640, stride=None):
        self.pyramid_levels = pyramid_levels
        self.stride = stride or [2 ** x for x in pyramid_levels]
        self.image_size = image_size

        # Scales & ratios dari YAML
        self.scales = scales or [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.ratios = ratios or [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def generate_anchors(self, base_size, scales, ratios):

        num_anchors = len(scales) * len(ratios)
        anchors = torch.zeros((num_anchors, 4))

        index = 0
        for scale, (ratio_w, ratio_h) in itertools.product(scales, ratios):
            area = base_size ** 2 * scale ** 2
            w = math.sqrt(area * ratio_w / ratio_h)
            h = math.sqrt(area * ratio_h / ratio_w)

            anchors[index, 0:4] = torch.tensor([-w / 2, -h / 2, w / 2, h / 2])
            index += 1

        return anchors

    def generate(self, device):
        all_anchors = []

        for i, p in enumerate(self.pyramid_levels):
            stride = self.stride[i]
            base_size = stride

            fm_size = int(math.ceil(self.image_size / stride))
            shifts_x = torch.arange(0, fm_size * stride, step=stride)
            shifts_y = torch.arange(0, fm_size * stride, step=stride)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)

            anchors = self.generate_anchors(base_size, self.scales, self.ratios).to(device)
            anchors = anchors[None, :, :] + shifts[:, None, :]  # broadcast
            anchors = anchors.reshape(-1, 4)

            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)
