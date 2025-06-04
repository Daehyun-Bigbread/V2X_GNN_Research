#!/usr/bin/env python3
"""
merge_v2x.py

data 디렉터리(예: data/8월, data/9월 등) 아래의 월별 한글 디렉터리 → 날짜(숫자) 디렉터리
→ 위치(C 또는 S) → 시간대(F/A/N/D) → 차량ID 구조에서
원천 CSV/JSON 데이터를 날짜+시간대(period) 기준으로 병합하여
`data/daily_merged/{month}/{date}_{location}_{period}_raw.csv`,
`data/daily_merged/{month}/{date}_{location}_{period}_raw.jsonl`,
`data/daily_merged/{month}/{date}_{location}_{period}_label.jsonl` 파일로 저장합니다.

Usage:
    python merge_v2x.py \
      --input-dir data \
      --output-dir data/daily_merged \
      --location C \
      --chunksize 500000
"""
import argparse
import pandas as pd
import json
from pathlib import Path

def merge_v2x(root_dir: Path, out_dir: Path, location: str, chunksize: int):
    root = Path(root_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 월별(한글) 디렉터리 순회: '8월', '9월', ...
    for month_dir in sorted(root.iterdir()):
        if not month_dir.is_dir():
            continue
        month = month_dir.name
        # 월별 출력 디렉터리 생성
        out_month = out_root / month
        out_month.mkdir(parents=True, exist_ok=True)

        # 날짜(숫자) 디렉터리: '220801', '220802', ...
        for date_dir in sorted(month_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name  # yymmdd 형식

            # 위치(C or S) 디렉터리
            loc_dir = date_dir / location
            if not loc_dir.exists():
                continue

            # 시간대(F/A/N/D) 디렉터리 순회
            for period_dir in sorted(loc_dir.iterdir()):
                if not period_dir.is_dir():
                    continue
                period = period_dir.name  # 'F', 'A', 'N', 'D'

                # 차량ID 디렉터리 순회
                for veh_dir in sorted(period_dir.iterdir()):
                    if not veh_dir.is_dir():
                        continue

                    # 각 차량 디렉터리 내 파일(.csv, .json) 순회
                    for data_file in sorted(veh_dir.glob('*')):
                        suffix = data_file.suffix.lower()

                        # CSV 원천 데이터 병합
                        if suffix == '.csv':
                            csv_out   = out_month / f"{date}_{location}_{period}_raw.csv"
                            jsonl_out = out_month / f"{date}_{location}_{period}_raw.jsonl"
                            first = not csv_out.exists()
                            for chunk in pd.read_csv(data_file, chunksize=chunksize):
                                # 시간대 정보를 컬럼에 추가
                                chunk['period'] = period
                                chunk.to_csv(csv_out, mode='a', header=first, index=False)
                                with open(jsonl_out, 'a', encoding='utf-8') as fj:
                                    for rec in chunk.to_dict(orient='records'):
                                        fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                first = False
                            print(f"[MERGE] {data_file} -> {csv_out.relative_to(Path.cwd())} (+ JSONL)")

                        # JSON 라벨 파일 병합
                        elif suffix == '.json':
                            label_out = out_month / f"{date}_{location}_{period}_label.jsonl"
                            ann = json.load(open(data_file, 'r', encoding='utf-8'))
                            ann['period'] = period
                            with open(label_out, 'a', encoding='utf-8') as fj:
                                fj.write(json.dumps(ann, ensure_ascii=False) + "\n")
                            print(f"[LABEL] {data_file} -> {label_out.relative_to(Path.cwd())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='월→날짜→위치→시간대→차량ID 구조 V2X 데이터 일별 병합 (period 포함)'
    )
    parser.add_argument('--input-dir',  required=True, help='data 루트 디렉터리')
    parser.add_argument('--output-dir', required=True, help='병합 결과 디렉터리')
    parser.add_argument('--location',   default='C', help='위치 ID (C 또는 S)')
    parser.add_argument('--chunksize',  type=int, default=500_000, help='CSV 청크 크기')
    args = parser.parse_args()

    merge_v2x(
        root_dir=Path(args.input_dir),
        out_dir=Path(args.output_dir),
        location=args.location,
        chunksize=args.chunksize
    )
