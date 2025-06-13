import argparse
import datetime
import os
import csv
import time
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from efficientdet.efficientdet import PostProcess   
from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def evaluate(model, val_generator, postprocess, params, config=None, input_size=640):
    model.eval()
    results = []


    coco_gt_path = os.path.join(opt.data_path, params.project_name, 'annotations', f'instances_{params.val_set}.json')
    coco_gt = COCO(coco_gt_path)
    image_ids = {img['file_name']: img['id'] for img in coco_gt.dataset['images']}

    category_ids_from_coco = sorted(coco_gt.getCatIds())
    coco_to_model_map = {coco_id: params.obj_list.index(cat['name']) 
                         for coco_id, cat in coco_gt.cats.items()}
    model_to_coco_map = {v: k for k, v in coco_to_model_map.items()}


    for data in tqdm(val_generator, desc='Evaluating'):
        with torch.no_grad():
            imgs = data['img'].cuda()
            scales = data['scale']
            filenames = data['img_name']

            # Langkah 1: Dapatkan output mentah dari model
            _, regression, classification, anchors = model.model(imgs)

            # Langkah 2: Panggil 'alat' postprocess terpisah
            scores, labels, boxes = postprocess(regression, classification, anchors, imgs)

            # Langkah 3: Format hasil untuk evaluasi COCO
            for i in range(len(boxes)):
                if filenames[i] not in image_ids: continue
                image_id = image_ids[filenames[i]]
                boxes[i] /= scales[i]

                for box, score, label in zip(boxes[i], scores[i], labels[i]):
                    # Gunakan mapping untuk mendapatkan ID kategori COCO yang benar
                    category_id = model_to_coco_map[label.item()]
                    result = {
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        'score': float(score)
                    }
                    results.append(result)

    if not results:
        print("No detections were made.")
        return {'mAP': 0, 'mAP50': 0, 'mAP75': 0, 'AR@1': 0, 'AR@10': 0, 'AR@100': 0}

    # Menjalankan COCOeval (kode ini sudah benar)
    import json, tempfile
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
        json.dump(results, f)
        pred_path = f.name
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.remove(pred_path)

    return {
        'mAP': coco_eval.stats[0], 'mAP50': coco_eval.stats[1], 'mAP75': coco_eval.stats[2],
        'AR@1': coco_eval.stats[6], 'AR@10': coco_eval.stats[7], 'AR@100': coco_eval.stats[8]
    }


def log_to_csv(epoch, train_box_loss, train_cls_loss, val_box_loss, val_cls_loss, mAP50, mAP, mAP75, AR_1, AR_10, AR_100, elapsed_time):
    os.makedirs('results', exist_ok=True)
    log_path = 'results/EfficientDETresults.csv'
    is_new = not os.path.exists(log_path)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                'epoch',
                'elapsed_seconds',
                'train/box_loss',
                'train/cls_loss',
                'val/box_loss',
                'val/cls_loss',
                'metrics/mAP50',
                'metrics/mAP',
                'metrics/mAP75',
                'metrics/AR@1',
                'metrics/AR@10',
                'metrics/AR@100'
            ])

        writer.writerow([
            epoch,
            round(elapsed_time, 2),
            train_box_loss,
            train_cls_loss,
            val_box_loss,
            val_cls_loss,
            mAP50,
            mAP,
            mAP75,
            AR_1,
            AR_10,
            AR_100 
        ])

def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    
    postprocess = PostProcess()
    
    config = {
        'num_classes': len(params.obj_list),
        'image_size': 640, 
        'soft_nms': False,
        'max_detection_points': 5000,
        'nms_threshold': 0.45,
        'confidence_threshold': 0.25, 
        'head_bn_level_first': False,
        'legacy_fpn': False,
    }
    

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    if params.num_gpus > 0:
        model = model.cuda()
        model = ModelWithLoss(model, debug=opt.debug)
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)
    else:
        model = ModelWithLoss(model, debug=opt.debug)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()
    start_time = time.time()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            train_cls_losses = []
            train_reg_losses = []
            epoch_loss = []

            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    optimizer.step()

                    train_cls_losses.append(cls_loss.item())
                    train_reg_losses.append(reg_loss.item())

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', opt.saved_path)
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                with torch.no_grad():
                    coco_stats = evaluate(model, val_generator, postprocess, params, 
                                  input_size=input_sizes[opt.compound_coef])
                model.train()

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                log_to_csv(
                    epoch=epoch,
                    train_box_loss=np.mean(train_reg_losses),
                    train_cls_loss=np.mean(train_cls_losses),
                    val_box_loss=reg_loss,
                    val_cls_loss=cls_loss,
                    mAP50=coco_stats['mAP50'],
                    mAP=coco_stats['mAP'],
                    mAP75=coco_stats['mAP75'],
                    elapsed_time=time.time() - start_time,
                    AR_1=coco_stats['AR@1'],
                    AR_10=coco_stats['AR@10'],
                    AR_100=coco_stats['AR@100'],
                )

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', opt.saved_path)

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', opt.saved_path)
        writer.close()
    writer.close()


def save_checkpoint(model, name, saved_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)