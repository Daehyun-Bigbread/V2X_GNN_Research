#!/usr/bin/env python3
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# CSV → Parquet 변환 스크립트 (위경도 포함)

def process_csv_file(csv_path: Path, out_dir: Path, chunksize: int):
    # 파일명에서 date, location 파싱
    date, loc = csv_path.stem.split('_')[:2]
    out_part = out_dir / f"date={date}" / f"location={loc}"
    out_part.mkdir(parents=True, exist_ok=True)
    out_file = out_part / f"{csv_path.stem}.parquet"

    writer = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # 1) timestamp 변환
        chunk['timestamp'] = pd.to_datetime(
            chunk['ISSUE_DATE'], format='%Y%m%d%H%M%S', errors='coerce'
        )
        # 2) 메타컬럼(date, location)
        chunk['date']     = date
        chunk['location'] = loc
        # 3) 위경도 포함, 필요한 컬럼만 선택
        cols = [
            'timestamp',
            'VEHICLE_ID',
            'LONGITUDE',
            'LATITUDE',
            'SPEED',
            'ACC_SEC',
            'BRAKE_STATUS',
            'period',
            'date',
            'location'
        ]
        sub = chunk[cols]

        # 4) PyArrow Table 변환
        table = pa.Table.from_pandas(sub)

        # 5) 첫 청크에서 ParquetWriter 생성
        if writer is None:
            writer = pq.ParquetWriter(
                str(out_file),
                table.schema,
                compression='snappy'
            )
        # 6) 청크를 파일에 기록
        writer.write_table(table)

    # Writer 닫기
    if writer:
        writer.close()
        print(f"[DONE] wrote {out_file}")
    else:
        print(f"[WARN] no data for {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='일별 병합 CSV를 Parquet으로 변환 (위경도 포함)'
    )
    parser.add_argument('--input-dir',  required=True, help='병합된 CSV들이 있는 루트 디렉터리')
    parser.add_argument('--output-dir', required=True, help='Parquet 저장 디렉터리')
    parser.add_argument('--pattern',    default='*_C_raw.csv', help='처리할 파일 패턴')
    parser.add_argument('--chunksize',  type=int, default=500_000, help='CSV 청크 크기')
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for f in sorted(inp.rglob(args.pattern)):
        process_csv_file(f, out, args.chunksize)