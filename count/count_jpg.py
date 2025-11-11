# count_jpgs.py
import os
import sys
from typing import Iterable

EXTS = {".jpg", ".jpeg"}  # 대소문자 무시

def count_jpgs(root: str) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext.lower() in EXTS:
                total += 1
    return total

def main(args: Iterable[str]) -> None:
    if len(args) < 2:
        print("사용법: python count_jpgs.py <경로>")
        sys.exit(1)
    root = args[1]
    if not os.path.isdir(root):
        print(f"디렉토리를 찾을 수 없습니다: {root}")
        sys.exit(1)

    total = count_jpgs(root)
    print(f"경로: {os.path.abspath(root)}")
    print(f".jpg/.jpeg 파일 개수: {total}")

if __name__ == "__main__":
    main(sys.argv)
