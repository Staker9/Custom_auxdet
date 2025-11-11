# -*- coding: utf-8 -*-
"""
로컬 단일 스크립트 학습 (argparse/외부 cfg 없이)
- 백본: ResNetV1cLocal (local_resnet.py)
- 넥  : AuxFPN (aux_fpn.py)
- RPN : RPNHead (rpn_head.py)  ※ local_layers의 ConvModule, batched_nms 사용
- RoI : SingleRoIExtractor + StandardRoIHead (single_level_roi_extractor.py / standard_roi_head.py)
- 데이터: COCO bbox 형식 (train/val)
필요 패키지: torch, torchvision, mmcv==2.*, mmdet==3.*, mmengine==0.10.*, pycocotools
"""

import os, sys, time, math, random
from typing import List, Dict, Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np

# --- 로컬 모듈 경로 추가 (현재 디렉토리 기반) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

# 네가 올린 로컬 모듈
from local_resnet import ResNetV1cLocal
from aux_fpn import AuxFPN
from rpn_head import RPNHead
from single_level_roi_extractor import SingleRoIExtractor
from standard_roi_head import StandardRoIHead
from local_layers import batched_nms  # (내부에서 rpn_head도 이미 사용)

# mmdet/mmengine 구성요소들 (레지스트리/러너 없이 직접 호출)
from local_mmdet.models.task_modules.prior_generators import AnchorGenerator
from local_mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder
from local_mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from local_mmdet.models.task_modules.assigners import MaxIoUAssigner
from local_mmdet.models.task_modules.samplers import RandomSampler
from local_mmdet.structures.bbox import bbox2roi
from local_mmdet.evaluation.functional import bbox_overlaps

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 1) COCO Dataset (간단 버전)
# =========================
class CocoLite(Dataset):
    def __init__(self, img_root, ann_file, classes: Tuple[str, ...], resize=(1333, 800), train=True):
        self.img_root = img_root
        self.ann = json.load(open(ann_file, 'r', encoding='utf-8'))
        self.train = train
        self.resize = resize
        self.classes = classes
        # id->file, id->anns
        self.id2img = {im['id']: im for im in self.ann['images']}
        self.img_ids = list(self.id2img.keys())
        self.cat_name2id = {c['name']: c['id'] for c in self.ann['categories']}
        self.cat_id2cls = {c['id']: i for i, c in enumerate(self.ann['categories'])}

        self.img2anns = {img_id: [] for img_id in self.img_ids}
        for a in self.ann['annotations']:
            if a.get('iscrowd', 0):  # crowd 제외
                continue
            self.img2anns[a['image_id']].append(a)

    def __len__(self):
        return len(self.img_ids)

    def _resize_keep_ratio(self, img: Image.Image, boxes: np.ndarray):
        w, h = img.size
        target_w, target_h = self.resize
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # pad to (target_w, target_h)
        pad_img = Image.new('RGB', (target_w, target_h), (114, 114, 114))
        pad_img.paste(img, (0, 0))
        if boxes.size > 0:
            boxes = boxes * scale
        return pad_img, boxes, scale

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        iminfo = self.id2img[img_id]
        path = os.path.join(self.img_root, iminfo['file_name'])
        img = Image.open(path).convert('RGB')

        anns = self.img2anns[img_id]
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id2cls[a['category_id']])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), np.int64)

        # Resize & pad
        img, boxes, scale = self._resize_keep_ratio(img, boxes)

        # To tensor & normalize
        img = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([123.675, 116.28, 103.53]) / 255.0
        std  = np.array([58.395, 57.12, 57.375]) / 255.0
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CHW

        target = {
            'bboxes': torch.from_numpy(boxes),   # (N,4) xyxy
            'labels': torch.from_numpy(labels),  # (N,)
            'scale': scale,
            'img_shape': (img.shape[1], img.shape[2])  # (H, W)
        }
        return img, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)

# =========================
# 2) Model Compose (로컬)
# =========================
class LocalFasterRCNN(nn.Module):
    """
    - backbone -> neck -> RPN -> proposals -> RoI -> bbox head
    - RPN/RCNN assigner+sampler는 mmdet의 모듈을 직접 사용
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # ---- Backbone & Neck ----
        self.backbone = ResNetV1cLocal(depth=50, in_channels=3,
                                       stem_channels=64, base_channels=64,
                                       out_indices=(0,1,2,3),
                                       norm_cfg=dict(type='BN'),
                                       act_cfg=dict(type='ReLU'))
        self.neck = AuxFPN(in_channels=[256,512,1024,2048],
                           out_channels=256, num_outs=2,
                           norm_cfg=dict(type='BN'),
                           act_cfg=dict(type='ReLU'))

        # ---- RPN ----
        self.anchor_generator = AnchorGenerator(
            strides=[4, 8], ratios=[0.5, 1.0, 2.0], scales=[8]
        )
        self.rpn_bbox_coder = DeltaXYWHBBoxCoder(
            target_means=[0.,0.,0.,0.], target_stds=[1.0,1.0,1.0,1.0]
        )
        self.rpn_head = RPNHead(
            in_channels=256,
            feat_channels=256,
            anchor_generator=self.anchor_generator,
            bbox_coder=self.rpn_bbox_coder,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            train_cfg=dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                              min_pos_iou=0.3, match_low_quality=True),
                sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                             neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=-1, pos_weight=-1, debug=False
            ),
            test_cfg=dict(
                nms_pre=1000, max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0
            )
        )

        # ---- RoI & BBox Head ----
        self.roi_extractor = SingleRoIExtractor(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4,8], finest_scale=56
        )
        self.bbox_head = Shared2FCBBoxHead(
            in_channels=256, fc_out_channels=1024, roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        )
        # StandardRoIHead를 그대로 써도 되지만(샘플러/어사이너 포함),
        # 여기서는 간단화를 위해 본 클래스에서 RCNN 단계만 직접 호출.

        # RCNN assigner/sampler
        self.rcnn_assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.0)
        self.rcnn_sampler  = RandomSampler(num=512, pos_fraction=0.25,
                                           neg_pos_ub=-1, add_gt_as_proposals=True)

    @torch.no_grad()
    def _get_rpn_proposals(self, feats: Tuple[torch.Tensor, ...], img_meta: Dict) -> torch.Tensor:
        """RPN 추론으로 proposals 생성 (단일 이미지 기준).
        반환: (M, 5) [img_id, x1,y1,x2,y2]
        """
        cls_scores, bbox_preds = self.rpn_head(feats)  # 각 level: (N,C,H,W)
        mlvl_priors = self.rpn_head.prior_generator.grid_priors(
            [s.shape[-2:] for s in cls_scores],
            device=cls_scores[0].device, with_stride=True
        )

        # single image만 처리 (N=1 가정)
        cls_list = [s[0] for s in cls_scores]
        reg_list = [b[0] for b in bbox_preds]
        results = self.rpn_head._predict_by_feat_single(
            cls_list, reg_list, score_factor_list=[None]*len(cls_list),
            mlvl_priors=mlvl_priors,
            img_meta=img_meta,
            cfg=None, rescale=False, with_nms=True
        )
        # results.bboxes: (K,4) xyxy
        img_inds = torch.zeros((results.bboxes.size(0), 1), device=results.bboxes.device)
        rois = torch.cat([img_inds, results.bboxes], dim=1)
        return rois

    def forward(self, x):
        # inference path (원한다면 구현) — 여기선 학습에서만 사용
        raise NotImplementedError

    def compute_losses_single(self, img: torch.Tensor, target: Dict) -> Dict[str, torch.Tensor]:
        """단일 배치(N=1) 기준 간소화 손실 계산(데모용).
        실제 사용에선 배치>1 케이스와 pad/collate에 맞춰 확장하면 됨.
        """
        img = img.unsqueeze(0)  # (1,C,H,W)
        feats_c = self.backbone(img)
        feats_p = self.neck(feats_c)  # tuple of [P4, P8] 등

        # ---- RPN losses ----
        gt_instances = dict(bboxes=target['bboxes'].to(img.device),
                            labels=torch.zeros((target['bboxes'].shape[0],), dtype=torch.long, device=img.device))  # RPN은 class-agnostic
        img_meta = dict(img_shape=(target['img_shape'][0], target['img_shape'][1], 3),
                        scale_factor=(1.0,1.0,1.0,1.0))
        cls_scores, bbox_preds, rpn_losses = self._rpn_loss(feats_p, [gt_instances], [img_meta])

        # ---- Proposals ----
        with torch.no_grad():
            rois = self._get_rpn_proposals(feats_p, img_meta)  # (M,5)

        if rois.shape[0] == 0:
            return {'loss_rpn': rpn_losses['loss_rpn_cls'] + rpn_losses['loss_rpn_bbox'],
                    'loss_rcnn': torch.tensor(0., device=img.device)}

        # ---- RCNN assign/sample ----
        # 간단화를 위해 IoU로 pos/neg split + 샘플링
        ious = bbox_overlaps(rois[:,1:], gt_instances['bboxes'])
        max_iou, argmax = ious.max(dim=1)
        pos_inds = torch.nonzero(max_iou >= 0.5, as_tuple=False).squeeze(1)
        neg_inds = torch.nonzero((max_iou < 0.5) & (max_iou >= 0.0), as_tuple=False).squeeze(1)

        num_pos = min(pos_inds.numel(), 128)
        num_neg = 512 - num_pos
        if pos_inds.numel() > 0:
            pos_inds = pos_inds[torch.randperm(pos_inds.numel(), device=img.device)[:num_pos]]
        if neg_inds.numel() > 0:
            neg_inds = neg_inds[torch.randperm(neg_inds.numel(), device=img.device)[:num_neg]]
        keep = torch.cat([pos_inds, neg_inds], dim=0) if (pos_inds.numel()+neg_inds.numel())>0 else pos_inds

        if keep.numel() == 0:
            return {'loss_rpn': rpn_losses['loss_rpn_cls'] + rpn_losses['loss_rpn_bbox'],
                    'loss_rcnn': torch.tensor(0., device=img.device)}

        rois_sampled = rois[keep]
        gt_inds = argmax[keep]
        gt_boxes = gt_instances['bboxes'][gt_inds]

        # label target
        labels = torch.full((keep.numel(),), fill_value=self.bbox_head.num_classes,  # background id
                            dtype=torch.long, device=img.device)
        if pos_inds.numel() > 0:
            labels[:pos_inds.numel()] = target['labels'][gt_inds[:pos_inds.numel()]]

        # ---- RoI features & bbox head ----
        roi_feats = self.roi_extractor(feats_p, rois_sampled)
        cls_score, bbox_pred = self.bbox_head(roi_feats)

        # ---- RCNN loss ----
        # classification
        loss_cls = nn.CrossEntropyLoss()(cls_score, labels)

        # bbox regression (class-specific)
        # bbox_pred: (N, num_classes*4)
        # gt target 만들기
        with torch.no_grad():
            # encode
            # rois_sampled[:,1:] vs gt_boxes
            targets = DeltaXYWHBBoxCoder(
                target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]
            ).encode(rois_sampled[:,1:], gt_boxes)

        # pos만 회귀
        if pos_inds.numel() > 0:
            pos_labels = labels[:pos_inds.numel()]
            pos_pred = bbox_pred[:pos_inds.numel()]
            # class-specific slice
            rows = torch.arange(pos_pred.size(0), device=img.device)
            col_start = (pos_labels * 4)
            # 안전장치: 배경 레이블은 회귀 제외
            valid = (pos_labels >= 0) & (pos_labels < self.bbox_head.num_classes)
            pos_rows = rows[valid]
            pos_cols = col_start[valid]
            pred_reg = torch.stack([
                pos_pred[pos_rows, pos_cols + 0],
                pos_pred[pos_rows, pos_cols + 1],
                pos_pred[pos_rows, pos_cols + 2],
                pos_pred[pos_rows, pos_cols + 3],
            ], dim=1)
            target_reg = targets[:pos_inds.numel()][valid]
            loss_bbox = nn.L1Loss()(pred_reg, target_reg)
        else:
            loss_bbox = torch.tensor(0., device=img.device)

        losses = {
            'loss_rpn': rpn_losses['loss_rpn_cls'] + rpn_losses['loss_rpn_bbox'],
            'loss_rcnn': loss_cls + loss_bbox
        }
        return losses

    def _rpn_loss(self, feats, batch_gt_instances, batch_img_metas):
        """RPNHead의 loss_by_feat을 직접 호출하기 위해 앵커/프라이어 생성 후 전달."""
        cls_scores, bbox_preds = self.rpn_head(feats)
        losses = self.rpn_head.loss(
            cls_scores, bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas
        )
        return cls_scores, bbox_preds, losses


# =========================
# 3) Train Loop (하드코딩)
# =========================
def main():
    # ---- 경로/클래스 세팅 ----
    CLASSES = ('cat', 'dog')  # ★ 네 데이터 클래스명으로 변경
    NUM_CLASSES = len(CLASSES)

    data_root = 'data/mydata'
    train_imgs = os.path.join(data_root, 'images/train')
    val_imgs   = os.path.join(data_root, 'images/val')
    train_ann  = os.path.join(data_root, 'annotations/instances_train.json')
    val_ann    = os.path.join(data_root, 'annotations/instances_val.json')

    # ---- Dataset/Dataloader ----
    train_set = CocoLite(train_imgs, train_ann, CLASSES, resize=(1333, 800), train=True)
    val_set   = CocoLite(val_imgs, val_ann, CLASSES, resize=(1333, 800), train=False)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                              num_workers=2, collate_fn=collate_fn, pin_memory=True)

    # ---- Model/Optim ----
    model = LocalFasterRCNN(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

    EPOCHS = 12
    SAVE_DIR = './work_dirs/local_run'
    os.makedirs(SAVE_DIR, exist_ok=True)

    model.train()
    global_step = 0
    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        for imgs, targets in train_loader:
            # 단순화를 위해 batch_size=1 기준
            img = imgs[0].to(DEVICE)
            target = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in targets[0].items()}

            optimizer.zero_grad()
            losses = model.compute_losses_single(img, target)
            loss = losses['loss_rpn'] + losses['loss_rcnn']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % 50 == 0:
                print(f"[ep {epoch:02d}] step {global_step:06d} "
                      f"loss={loss.item():.4f}  (rpn={losses['loss_rpn']:.4f}, rcnn={losses['loss_rcnn']:.4f})")

        print(f"[ep {epoch:02d}] epoch_loss={epoch_loss/len(train_loader):.4f}")
        # ckpt 저장
        torch.save({'model': model.state_dict(), 'epoch': epoch},
                   os.path.join(SAVE_DIR, f'epoch_{epoch:02d}.pt'))

    print("Done.")

if __name__ == "__main__":
    main()
