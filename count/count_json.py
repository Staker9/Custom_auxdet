# count_file_name_in_json.py
import json
import argparse
import os
from typing import Any, Set, Tuple

def iter_file_name_values(node: Any):
    """JSON 객체(딕셔너리/리스트/스칼라)를 재귀 순회하며
    key == 'file_name' 인 값들을 yield."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "file_name":
                yield v
            # 하위 구조 재귀 탐색
            yield from iter_file_name_values(v)
    elif isinstance(node, list):
        for item in node:
            yield from iter_file_name_values(item)
    # 스칼라 타입은 무시

def verify_json_consumed_fully(text: str) -> Tuple[bool, int, str]:
    """
    JSON 문자열을 raw_decode로 파싱하여 끝까지 소비되었는지 확인.
    반환: (is_fully_consumed, end_char_index, non_ws_trailer_preview)
    - is_fully_consumed: 공백 이외의 잔여 문자가 없으면 True
    - end_char_index: JSON 파싱이 끝난 문자 오프셋(0-based, 파이썬 문자열 인덱스)
    - non_ws_trailer_preview: 잔여 비공백 문자가 있을 경우 앞부분 미리보기(최대 80자)
    """
    decoder = json.JSONDecoder()
    obj, end_idx = decoder.raw_decode(text)
    # end_idx 이후가 모두 공백이면 OK
    trailer = text[end_idx:]
    non_ws = trailer.strip()
    if non_ws:
        return (False, end_idx, non_ws[:80])
    return (True, end_idx, "")

def main():
    ap = argparse.ArgumentParser(description="JSON에서 'file_name' 키 개수 세기 + 파일 끝까지 읽었는지 검증")
    ap.add_argument("json_path", help="입력 JSON 파일 경로")
    ap.add_argument("--unique", action="store_true",
                    help="서로 다른 file_name 값의 개수도 출력")
    ap.add_argument("--show-trailer", type=int, default=120,
                    help="끝부분(트레일러) 원문 미리보기 최대 문자 수 (기본 120)")
    args = ap.parse_args()

    # 파일 전체 텍스트를 읽고 검증
    file_size_bytes = os.path.getsize(args.json_path)
    with open(args.json_path, "r", encoding="utf-8") as f:
        text = f.read()
        # 파일 포인터는 EOF여야 함
        tell_after_read = f.tell()

    # JSON 파싱 + 끝까지 소비되었는지 확인
    try:
        fully, end_char_idx, trailer_preview = verify_json_consumed_fully(text)
    except json.JSONDecodeError as e:
        print("[ERROR] JSON 파싱 실패:", str(e))
        print(" - 에러 위치(문자 오프셋):", getattr(e, "pos", "?"))
        print(" - 에러 줄/열:", getattr(e, "lineno", "?"), getattr(e, "colno", "?"))
        return

    # 실제 객체 로드(성능 위해 두 번 파싱하기 싫으면 위 decoder 결과 obj를 활용해도 됨)
    data = json.loads(text)

    # 개수 계산
    values = list(iter_file_name_values(data))
    total = len(values)

    print("=== file_name 카운트 ===")
    print(f"총 'file_name' 키 개수: {total}")
    if args.unique:
        uniq: Set[Any] = set(values)
        print(f"서로 다른 'file_name' 값 개수: {len(uniq)}")

    # 파일 끝까지 읽은 것 검증 리포트
    print("\n=== JSON 소비 검증 ===")
    print(f"- 파일 크기(바이트): {file_size_bytes}")
    # 문자 인덱스를 바이트로 환산(UTF-8 가변 길이이므로 근사값 대신, 원문 슬라이스를 인코딩해서 정확히 계산)
    consumed_bytes = len(text[:end_char_idx].encode('utf-8'))
    print(f"- JSON 파싱이 끝난 위치(문자 인덱스): {end_char_idx}")
    print(f"- JSON 파싱이 끝난 위치(바이트 환산): {consumed_bytes}")
    print(f"- 파일을 전부 읽었을 때 포인터 위치 f.tell(): {tell_after_read}")

    if fully:
        print("- 결과: ✅ 공백을 제외한 잔여 문자가 없습니다. (끝까지 소비됨)")
        # 끝부분 미리보기(참고용)
        tail_preview = text[max(0, end_char_idx - args.show_trailer): end_char_idx]
        print(f"- 끝부분 직전 미리보기({len(tail_preview)}자):")
        print(repr(tail_preview))
    else:
        print("- 결과: ⚠️ JSON 본문 이후에 공백 이외의 잔여 문자가 있습니다. (완전 소비 아님)")
        print(f"- 잔여 비공백 시작 미리보기({min(len(trailer_preview), args.show_trailer)}자):")
        print(repr(trailer_preview[:args.show_trailer]))

if __name__ == "__main__":
    main()
