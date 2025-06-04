#!/usr/bin/env python3
"""
add_period.py

기존 병합된 CSV 파일들에 대해 ISSUE_DATE 기반으로 시각(hour)을 파싱하여
시간대(period) 컬럼만 추가하는 스크립트

Usage:
  python add_period.py \
    --input-dir data/daily_merged \
    --chunksize 500000

설명:
  --input-dir 아래의 모든 하위 폴더에서 '*_raw.csv' 파일을 재귀적으로 찾아,
  ISSUE_DATE 컬럼을 기준으로 시간(hour)을 뽑아내어
  다음 매핑 규칙대로 'period' 컬럼을 추가합니다:

    06 <= hour < 12  → 'F' (오전)
    12 <= hour < 18  → 'A' (오후)
    18 <= hour < 24  → 'N' (밤)
    그 외             → 'D' (새벽)

  큰 파일도 처리할 수 있도록 청크 단위(chunked)로 읽고 덮어씁니다.
"""
import argparse
import pandas as pd
from pathlib import Path

def map_period(hour: int) -> str:
    if 6 <= hour < 12:
        return 'F'
    if 12 <= hour < 18:
        return 'A'
    if 18 <= hour < 24:
        return 'N'
    return 'D'


def add_period_column(input_dir: Path, chunksize: int):
    csv_paths = list(input_dir.rglob('*_raw.csv'))
    if not csv_paths:
        print(f"[WARN] 처리할 CSV 파일을 찾을 수 없습니다: {input_dir}")
        return

    for csv_path in csv_paths:
        print(f"[PROCESS] {csv_path}")
        temp_path = csv_path.with_suffix('.tmp.csv')
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            # ISSUE_DATE 문자열 -> datetime -> hour
            chunk['hour'] = pd.to_datetime(
                chunk['ISSUE_DATE'], format='%Y%m%d%H%M%S', errors='coerce'
            ).dt.hour
            chunk['period'] = chunk['hour'].apply(map_period)
            chunk.drop(columns=['hour'], inplace=True)

            # 청크별 덮어쓰기
            chunk.to_csv(
                temp_path,
                mode='a',
                header=first,
                index=False
            )
            first = False

        # 원본 덮어쓰기
        temp_path.replace(csv_path)
        print(f"[DONE] period 컬럼 추가 완료: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='기존 병합된 CSV에 period 컬럼을 ISSUE_DATE 기반으로 추가'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help='병합된 CSV들이 있는 최상위 폴더'
    )
    parser.add_argument(
        '--chunksize', type=int, default=500_000,
        help='pd.read_csv 청크 크기'
    )
    args = parser.parse_args()

    add_period_column(
        input_dir=Path(args.input_dir),
        chunksize=args.chunksize
    )
