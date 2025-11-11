# split_coco_images.py
# ------------------------------------------------------------
# COCO JSON에 기록된 이미지 목록을 기준으로
# data_dir 아래의 원본 이미지를 images/{split}/ 로 복사합니다.
# - YOLO 변환/라벨 생성 없음
# - Hydra, pycocotools 불필요 (json 직접 파싱)
# ------------------------------------------------------------
import os
import json
import shutil
from typing import Dict, Optional, Tuple

# ====== 사용자 설정 ======
DATA_DIR = r"C:\Users\IDAL\Desktop\WACV\frames"  # 원본 이미지들이 있는 루트 폴더
SPLIT_JSONS: Dict[str, str] = {
    "train": r"C:\Users\IDAL\Desktop\WACV\Train.json",
    "val":   r"C:\Users\IDAL\Desktop\WACV\Train.json",
    # 필요하면 주석 해제
    "test":  r"C:\Users\IDAL\Desktop\WACV\TestNoLabels.json",
}
OUTPUT_DIR: Optional[str] = None  # None이면 DATA_DIR/images 밑으로 복사
COPY_MODE = "copy"  # "copy" | "move" (원본 이동) | "link"(가능한 OS에서 하드링크 시도)
DRY_RUN = False     # True면 실제 복사/이동하지 않고 계획만 출력
# =======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def copy_one(src: str, dst: str, mode: str) -> None:
    if DRY_RUN:
        print(f"[DRY] {mode.upper():4} {src} -> {dst}")
        return

    ensure_dir(os.path.dirname(dst))
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "link":
        # 하드링크 시도 → 실패 시 복사로 폴백 (Windows 권한 이슈 대비)
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_image_list(json_path: str) -> Tuple[str, list]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # COCO: images = [{id, file_name, width, height, ...}, ...]
    images = data.get("images", [])
    return json_path, [img.get("file_name", "") for img in images]

def main():
    base_out = OUTPUT_DIR if OUTPUT_DIR else os.path.join(DATA_DIR, "images")
    ensure_dir(base_out)

    total_found = 0
    total_missing = 0

    for split, json_path in SPLIT_JSONS.items():
        if not json_path:
            continue

        out_dir = os.path.join(base_out, split)
        ensure_dir(out_dir)

        jp, file_names = load_image_list(json_path)
        print(f"\n[Split: {split}] JSON: {jp}")
        print(f"- 대상 이미지 수: {len(file_names)}")

        found, missing = 0, 0
        for fname in file_names:
            # COCO의 file_name이 하위 경로를 포함할 수도 있음
            src = os.path.join(DATA_DIR, fname)
            # 동일한 상대 경로 구조를 유지하고 싶다면 다음 줄 사용:
            # dst = os.path.join(out_dir, fname)
            # 파일명이 겹칠 수 있어 평면 폴더가 필요하면 다음처럼 저장:
            safe_name = fname.replace("/", "_").replace("\\", "_")
            dst = os.path.join(out_dir, safe_name)

            if os.path.exists(src):
                copy_one(src, dst, COPY_MODE)
                found += 1
            else:
                # DATA_DIR 바로 아래가 아닌 경우를 대비해 fallback 탐색(1단계)
                alt = os.path.join(DATA_DIR, os.path.basename(fname))
                if os.path.exists(alt):
                    copy_one(alt, dst, COPY_MODE)
                    found += 1
                else:
                    print(f"[WARN] 원본 없음: {src}")
                    missing += 1

        print(f"- 복사/이동 완료: {found}, 누락: {missing}, 출력 폴더: {out_dir}")
        total_found += found
        total_missing += missing

    print("\n[요약]")
    print(f"총 성공: {total_found}, 총 누락: {total_missing}")
    print(f"출력 루트: {base_out}")
    print(f"모드: {COPY_MODE} | DRY_RUN={DRY_RUN}")

if __name__ == "__main__":
    main()
