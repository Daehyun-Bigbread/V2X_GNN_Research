#!/usr/bin/env python3
# ISSUE_DATE 형식 확인 스크립트

import pandas as pd
import os

# 첫 번째 파일만 확인
data_dir = "data/daily_merged/08월"
files = [f for f in os.listdir(data_dir) if f.endswith('_raw.csv')]
first_file = os.path.join(data_dir, files[0])

print(f"📄 파일: {first_file}")

# 첫 몇 행만 읽기
df = pd.read_csv(first_file, nrows=10)

print("\n📊 컬럼 목록:")
print(df.columns.tolist())

print("\n📅 ISSUE_DATE 샘플:")
if 'ISSUE_DATE' in df.columns:
    print(df['ISSUE_DATE'].head())
    print(f"데이터 타입: {df['ISSUE_DATE'].dtype}")
    print(f"첫 번째 값: {df['ISSUE_DATE'].iloc[0]} (타입: {type(df['ISSUE_DATE'].iloc[0])})")
    
    # 파싱 테스트
    try:
        test_date = pd.to_datetime(df['ISSUE_DATE'].iloc[0], format='%Y%m%d%H%M%S')
        print(f"✅ 파싱 성공: {test_date}")
    except Exception as e:
        print(f"❌ 파싱 실패: {e}")
        
        # 다른 형식들 시도
        formats_to_try = [
            '%Y%m%d%H%M%S',  # 20220801164437
            '%Y-%m-%d %H:%M:%S',  # 2022-08-01 16:44:37
            '%Y/%m/%d %H:%M:%S',  # 2022/08/01 16:44:37
            '%Y%m%d_%H%M%S',  # 20220801_164437
        ]
        
        for fmt in formats_to_try:
            try:
                test_date = pd.to_datetime(str(df['ISSUE_DATE'].iloc[0]), format=fmt)
                print(f"✅ 형식 {fmt}로 파싱 성공: {test_date}")
                break
            except:
                print(f"❌ 형식 {fmt} 실패")
else:
    print("❌ ISSUE_DATE 컬럼이 없습니다!")

print("\n🔍 전체 첫 5행:")
print(df.head())