"""YOLO 객체 탐지 — 이미지 / 웹캠 지원, 결과를 파일로 저장한다.

사용법
  uv run detect.py --source samples/bus.jpg          # 이미지 한 장
  uv run detect.py --source 0                        # 웹캠 (s: 저장, q/ESC: 종료)

저장 결과 (outputs/ 폴더)
  <이름>_annotated.jpg  : 박스가 그려진 결과 이미지
  detections.csv        : 탐지된 객체 1건 = 1행 (클래스, 신뢰도, 박스 좌표, 확인 필요 여부)
  run_log.json          : 실행 요약 (모델, 장치, 탐지 개수, 탐지 0건 여부)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

REVIEW_CONF = 0.50  # 이 값보다 신뢰도가 낮으면 '확인 필요'로 표시한다


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 객체 탐지")
    p.add_argument("--source", default="0", help="이미지 경로 또는 카메라 번호(기본 0)")
    p.add_argument("--model", default="yolo11n.pt", help="모델 가중치 (없으면 자동 다운로드)")
    p.add_argument("--conf", type=float, default=0.25, help="탐지 최소 신뢰도")
    p.add_argument("--out", default="outputs", help="결과 저장 폴더")
    return p.parse_args()


def rows_from_result(result, source_name):
    """탐지 결과를 CSV 행 목록으로 바꾼다. 신뢰도가 낮은 행은 확인 필요로 표시한다."""
    rows = []
    for box in result.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
        rows.append({
            "source": source_name,
            "class": result.names[int(box.cls[0])],
            "confidence": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "needs_review": "예" if conf < REVIEW_CONF else "아니오",
        })
    return rows


def write_outputs(out_dir, stem, result, rows, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(img_path), result.plot())

    csv_path = out_dir / "detections.csv"
    new_file = not csv_path.exists()
    fields = ["source", "class", "confidence", "x1", "y1", "x2", "y2", "needs_review"]
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerows(rows)

    meta.update({
        "output_image": img_path.name,
        "detections": len(rows),
        "no_detection": len(rows) == 0,  # 탐지 0건이면 숨기지 않고 기록한다
        "needs_review_count": sum(r["needs_review"] == "예" for r in rows),
    })
    log_path = out_dir / "run_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    log.append(meta)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return img_path


def main():
    args = parse_args()
    out_dir = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "device": device,
        "conf_threshold": args.conf,
        "review_threshold": REVIEW_CONF,
    }

    if args.source.isdigit():  # 웹캠 모드
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            sys.exit(f"카메라 {args.source}번을 열 수 없습니다. 다른 번호로 시도: --source 1")
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model(frame, conf=args.conf, device=device, verbose=False)[0]
            cv2.imshow("YOLO webcam  (s: save, q: quit)", result.plot())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                saved += 1
                stem = f"webcam_{time.strftime('%H%M%S')}"
                rows = rows_from_result(result, stem + ".jpg")
                print("저장:", write_outputs(out_dir, stem, result, rows, dict(meta, source=f"camera:{args.source}")))
            elif key in (ord("q"), 27):
                break
        cap.release()
        cv2.destroyAllWindows()
        print(f"웹캠 종료 — 저장한 프레임 {saved}장")
    else:  # 이미지 모드
        src = Path(args.source)
        if not src.exists():
            sys.exit(f"이미지를 찾을 수 없습니다: {src}")
        result = model(str(src), conf=args.conf, device=device, verbose=False)[0]
        rows = rows_from_result(result, src.name)
        img = write_outputs(out_dir, src.stem, result, rows, dict(meta, source=src.name))
        print(f"{src.name}: {len(rows)}건 탐지 → {img}")
        for r in rows:
            print(f"  {r['class']:<12} {r['confidence']:.2f}  확인필요={r['needs_review']}")


if __name__ == "__main__":
    main()
