# local_layers.py
import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) ConfigDict: 간단 alias
# -----------------------------
ConfigDict = Dict[str, Any]

# -----------------------------
# 2) Minimal InstanceData
# - 속성(bboxes/scores/labels/level_ids 등) 동적 보관
# - 텐서 인덱싱/마스킹만 지원(본 파일에서 쓰는 범위)
# -----------------------------
class InstanceData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        # 첫 텐서 속성 길이를 기준으로
        for v in self.__dict__.values():
            if torch.is_tensor(v):
                return v.size(0)
        return 0

    def _index_value(self, v, idx):
        if torch.is_tensor(v):
            return v[idx]
        return v  # 텐서가 아니면 그대로 반환

    def __getitem__(self, idx):
        out = InstanceData()
        for k, v in self.__dict__.items():
            setattr(out, k, self._index_value(v, idx))
        return out

    def keys(self):
        return self.__dict__.keys()

    def __repr__(self):
        keys = ", ".join(list(self.__dict__.keys()))
        return f"InstanceData({keys})"

# -----------------------------
# 3) ConvModule (MMCV-lite)
# - conv -> norm -> act 순서
# - norm_cfg: {"type": "BN"} | {"type":"GN","num_groups":32} 지원
# - act_cfg: {"type":"ReLU"| "SiLU"|"GELU"}, inplace 옵션 지원
# - conv_cfg 무시는 기본 Conv2d 사용
# -----------------------------
def _build_norm(norm_cfg: Optional[ConfigDict], num_features: int):
    if not norm_cfg:
        return None
    t = norm_cfg.get("type", "BN")
    if t == "BN":
        return nn.BatchNorm2d(num_features)
    if t == "SyncBN":
        # 단일 GPU/로컬 환경에서는 일반 BN로 대체
        return nn.BatchNorm2d(num_features)
    if t == "GN":
        groups = norm_cfg.get("num_groups", 32)
        return nn.GroupNorm(groups, num_features)
    raise ValueError(f"Unsupported norm type: {t}")

def _build_act(act_cfg: Optional[ConfigDict]):
    if not act_cfg:
        return None
    t = act_cfg.get("type", "ReLU")
    inplace = act_cfg.get("inplace", True)
    if t == "ReLU":
        return nn.ReLU(inplace=inplace)
    if t == "SiLU":
        return nn.SiLU(inplace=inplace)
    if t == "GELU":
        # GELU에는 inplace 없음
        return nn.GELU()
    raise ValueError(f"Unsupported act type: {t}")

class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        conv_cfg: Optional[ConfigDict] = None,   # 무시
        norm_cfg: Optional[ConfigDict] = None,
        act_cfg: Optional[ConfigDict] = dict(type="ReLU"),
        inplace: bool = True,
    ):
        super().__init__()
        # bias는 norm 유무에 맞춰 자동 결정
        if bias is None:
            bias = norm_cfg is None

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.norm = _build_norm(norm_cfg, out_channels)
        # act_cfg가 있고 type만 지정되고 inplace 전달되면 반영
        if act_cfg and "inplace" not in act_cfg and act_cfg.get("type", "") == "ReLU":
            act_cfg = {**act_cfg, "inplace": inplace}
        self.act = _build_act(act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

# -----------------------------
# 4) batched_nms (torchvision 사용, 없으면 fallback)
# - cfg 예: {"type":"nms", "iou_threshold":0.7}만 지원
# -----------------------------
def _nms_torchvision(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float):
    from torchvision.ops import nms
    keep = nms(boxes, scores, iou_thr)
    return keep

def batched_nms(boxes: torch.Tensor,
                scores: torch.Tensor,
                idxs: torch.Tensor,
                nms_cfg: Optional[ConfigDict] = None):
    """
    Args:
      boxes: (N, 4) xyxy
      scores: (N,)
      idxs: (N,) 같은 값은 같은 그룹(레벨/클래스)으로 NMS 묶음
      nms_cfg: {"type": "nms", "iou_threshold": float}
    Returns:
      dets: (M, 5) [x1,y1,x2,y2,score]
      keep: (M,) 인덱스
    """
    if boxes.numel() == 0:
        return torch.zeros((0, 5), device=boxes.device, dtype=boxes.dtype), boxes.new_zeros((0,), dtype=torch.long)

    iou_thr = 0.5
    if nms_cfg is not None:
        if nms_cfg.get("type", "nms") != "nms":
            raise ValueError("Only 'nms' is supported in this lightweight batched_nms.")
        iou_thr = float(nms_cfg.get("iou_threshold", 0.5))

    keep_all = []
    unique_ids = torch.unique(idxs)
    for uid in unique_ids:
        mask = (idxs == uid)
        b = boxes[mask]
        s = scores[mask]
        if b.numel() == 0:
            continue
        try:
            kept = _nms_torchvision(b, s, iou_thr)
        except Exception:
            # 아주 단순한 fallback: 점수순 정렬 후 겹침 제거(느슨)
            order = torch.argsort(s, descending=True)
            kept = []
            while order.numel() > 0:
                i = order[0].item()
                kept.append(i)
                if order.numel() == 1:
                    break
                i_box = b[i].unsqueeze(0)
                rest = b[order[1:]]
                # IoU 계산
                x1 = torch.maximum(i_box[:, 0], rest[:, 0])
                y1 = torch.maximum(i_box[:, 1], rest[:, 1])
                x2 = torch.minimum(i_box[:, 2], rest[:, 2])
                y2 = torch.minimum(i_box[:, 3], rest[:, 3])
                inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                area_i = (i_box[:, 2]-i_box[:, 0]) * (i_box[:, 3]-i_box[:, 1])
                area_r = (rest[:, 2]-rest[:, 0]) * (rest[:, 3]-rest[:, 1])
                iou = inter / (area_i + area_r - inter + 1e-6)
                remain = (iou <= iou_thr).nonzero(as_tuple=False).squeeze(1)
                order = order[1:][remain]
            kept = torch.tensor(kept, device=b.device, dtype=torch.long)

        global_idx = mask.nonzero(as_tuple=False).squeeze(1)[kept]
        keep_all.append(global_idx)

    if len(keep_all) == 0:
        return torch.zeros((0, 5), device=boxes.device, dtype=boxes.dtype), boxes.new_zeros((0,), dtype=torch.long)

    keep = torch.cat(keep_all, dim=0)
    # 최종 det_bboxes는 (M, 5)
    dets = torch.cat([boxes[keep], scores[keep].unsqueeze(1)], dim=1)
    # 점수로 최종 정렬
    order = torch.argsort(dets[:, -1], descending=True)
    dets = dets[order]
    keep = keep[order]
    return dets, keep

class BaseModule(nn.Module):
    """
    mmcv/mmengine 없이 쓰는 초경량 BaseModule.
    - init_cfg는 받아서 보관만 하고, 기본 init_weights()에서 일반적인 초기화 수행
    - 필요하면 하위 클래스에서 init_weights() 오버라이드 가능
    """
    def __init__(self, init_cfg: Optional[ConfigDict] = None):
        super().__init__()
        self.init_cfg = init_cfg
        self._is_initialized = False
        self.init_weights()

    def init_weights(self):
        """기본 가중치 초기화 (Kaiming/BN=1,0/Linear=Normal(0,0.01))."""
        if self._is_initialized:
            return

        def _init(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)
        self._is_initialized = True