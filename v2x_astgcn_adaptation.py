#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2X 데이터를 광주 지역 그리드 기반 AST-GCN 이상탐지 형식으로 변환

핵심 수정사항:
- 광주광역시 지역에만 특화
- 위치 데이터 이상값 제거 및 광주 범위 제한
- 격자 수 최적화 (관리 가능한 수준)
- 이상탐지 로직 개선

Author: V2X Grid-based Anomaly Detection Team
Date: 2025-06-07
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 광주광역시 좌표 범위 정의
GWANGJU_BOUNDS = {
    'lon_min': 126.7, 'lon_max': 127.2,  # 경도 범위 (약 50km)
    'lat_min': 35.0, 'lat_max': 35.3      # 위도 범위 (약 30km)
}

def load_v2x_data(data_dir):
    """V2X 원본 데이터 로딩"""
    print(f"📂 V2X 이상탐지 데이터 로딩: {data_dir}")
    
    # 1. 모든 주행 데이터 로딩
    raw_files = [f for f in os.listdir(data_dir) if f.endswith('_raw.csv')]
    if not raw_files:
        raise FileNotFoundError(f"❌ {data_dir}에서 *_raw.csv 파일을 찾을 수 없습니다")
    
    print(f"   📄 발견된 주행 데이터 파일: {len(raw_files)}개")
    
    # 모든 파일 로딩 (전체 데이터 사용)
    all_raw_data = []
    max_files = len(raw_files)  # 모든 파일 로딩
    
    for i, raw_file in enumerate(sorted(raw_files)[:max_files]):
        file_path = os.path.join(data_dir, raw_file)
        try:
            df = pd.read_csv(file_path)
            all_raw_data.append(df)
            print(f"       ✅ {raw_file}: {df.shape}")
        except Exception as e:
            print(f"       ❌ {raw_file} 로딩 실패: {e}")
    
    # 모든 데이터 합치기
    if all_raw_data:
        raw_df = pd.concat(all_raw_data, ignore_index=True)
        print(f"   📊 통합 주행 데이터: {raw_df.shape}")
    else:
        raise ValueError("❌ 로딩된 주행 데이터가 없습니다")
    
    # 2. 해당하는 라벨 데이터 로딩
    label_files = [f for f in os.listdir(data_dir) if f.endswith('_label.jsonl')]
    label_data = []
    
    if label_files:
        print(f"   📄 발견된 라벨 데이터 파일: {len(label_files)}개")
        
        for i, label_file in enumerate(sorted(label_files)):
            file_path = os.path.join(data_dir, label_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_labels = []
                    for line in f:
                        file_labels.append(json.loads(line.strip()))
                    label_data.extend(file_labels)
                    print(f"       ✅ {label_file}: {len(file_labels)}개")
            except Exception as e:
                print(f"       ❌ {label_file} 로딩 실패: {e}")
        
        print(f"   📊 통합 라벨 데이터: {len(label_data)}개")
    else:
        print("   ⚠️ 라벨 데이터가 없습니다. 속도 기반 이상탐지로 진행합니다.")
    
    return raw_df, label_data

def clean_and_filter_location_data(raw_df):
    """
    🧹 위치 데이터 정제 및 광주 범위 필터링
    """
    print("🧹 위치 데이터 정제 중 (광주 지역 특화)...")
    
    initial_count = len(raw_df)
    bounds = GWANGJU_BOUNDS
    
    # 1. 기본 위치 데이터 확인
    if 'LONGITUDE' not in raw_df.columns or 'LATITUDE' not in raw_df.columns:
        print("   ⚠️ 위치 컬럼이 없어 광주 지역 중심으로 가상 위치를 생성합니다.")
        # 광주 지역 중심으로 가상 위치 생성
        num_records = len(raw_df)
        raw_df['LONGITUDE'] = np.random.uniform(bounds['lon_min'], bounds['lon_max'], num_records)
        raw_df['LATITUDE'] = np.random.uniform(bounds['lat_min'], bounds['lat_max'], num_records)
        print(f"   ✅ 광주 지역 가상 위치 생성: {num_records:,}개")
        return raw_df
    
    # 2. 이상값 제거
    print("   🔍 위치 데이터 이상값 검사...")
    
    # 기본 유효성 검사
    valid_lon = (raw_df['LONGITUDE'] >= -180) & (raw_df['LONGITUDE'] <= 180)
    valid_lat = (raw_df['LATITUDE'] >= -90) & (raw_df['LATITUDE'] <= 90)
    basic_valid = valid_lon & valid_lat & (raw_df['LONGITUDE'] != 0) & (raw_df['LATITUDE'] != 0)
    
    print(f"     기본 유효 레코드: {basic_valid.sum():,}개 ({basic_valid.mean()*100:.1f}%)")
    
    # 광주 지역 범위 필터링
    in_gwangju = (
        (raw_df['LONGITUDE'] >= bounds['lon_min']) & 
        (raw_df['LONGITUDE'] <= bounds['lon_max']) &
        (raw_df['LATITUDE'] >= bounds['lat_min']) & 
        (raw_df['LATITUDE'] <= bounds['lat_max'])
    )
    
    gwangju_count = in_gwangju.sum()
    print(f"     광주 지역 내 레코드: {gwangju_count:,}개 ({gwangju_count/initial_count*100:.1f}%)")
    
    # 3. 필터링 전략 결정
    if gwangju_count < initial_count * 0.1:  # 광주 지역 데이터가 10% 미만인 경우
        print("   ⚠️ 광주 지역 데이터가 부족하여 전체 데이터를 광주 범위로 재매핑합니다.")
        
        # 전체 데이터의 위치를 광주 범위로 정규화
        if basic_valid.sum() > 0:
            valid_df = raw_df[basic_valid].copy()
            
            # 원래 위치 데이터의 범위 계산
            orig_lon_min, orig_lon_max = valid_df['LONGITUDE'].min(), valid_df['LONGITUDE'].max()
            orig_lat_min, orig_lat_max = valid_df['LATITUDE'].min(), valid_df['LATITUDE'].max()
            
            # 광주 범위로 정규화
            lon_normalized = (valid_df['LONGITUDE'] - orig_lon_min) / (orig_lon_max - orig_lon_min)
            lat_normalized = (valid_df['LATITUDE'] - orig_lat_min) / (orig_lat_max - orig_lat_min)
            
            valid_df['LONGITUDE'] = bounds['lon_min'] + lon_normalized * (bounds['lon_max'] - bounds['lon_min'])
            valid_df['LATITUDE'] = bounds['lat_min'] + lat_normalized * (bounds['lat_max'] - bounds['lat_min'])
            
            print(f"   ✅ 위치 재매핑 완료: {len(valid_df):,}개 레코드")
            return valid_df
        else:
            # 완전히 새로운 가상 위치 생성
            print("   🆕 광주 지역 가상 위치 생성")
            raw_df['LONGITUDE'] = np.random.uniform(bounds['lon_min'], bounds['lon_max'], len(raw_df))
            raw_df['LATITUDE'] = np.random.uniform(bounds['lat_min'], bounds['lat_max'], len(raw_df))
            return raw_df
    else:
        # 광주 지역 데이터만 사용
        filtered_df = raw_df[basic_valid & in_gwangju].copy()
        print(f"   ✅ 광주 지역 필터링 완료: {len(filtered_df):,}개 레코드 유지")
        return filtered_df

def create_anomaly_labels(raw_df, label_data):
    """V2X 데이터에서 이상상황 라벨 생성 (개선된 로직)"""
    print("🚨 이상상황 라벨 생성 중...")
    
    # 차량 ID 매핑
    if 'VEHICLE_ID' in raw_df.columns:
        raw_df['TRIP_ID'] = raw_df['VEHICLE_ID']
    
    # 이상 점수 초기화
    raw_df['anomaly_score'] = 0.0
    
    # 1. 속도 기반 이상탐지 (더 엄격한 기준)
    print("   🚗 속도 기반 이상탐지...")
    
    # 저속 이상 (3km/h 미만 = 정체/사고) - 더 엄격하게
    low_speed_mask = raw_df['SPEED'] < 3
    raw_df.loc[low_speed_mask, 'anomaly_score'] += 0.3
    print(f"     저속 이상 (< 3km/h): {low_speed_mask.sum():,}건")
    
    # 고속 이상 (광주 시내 기준 70km/h 초과) - 더 엄격하게
    high_speed_mask = raw_df['SPEED'] > 70
    raw_df.loc[high_speed_mask, 'anomaly_score'] += 0.2
    print(f"     고속 이상 (> 70km/h): {high_speed_mask.sum():,}건")
    
    # 급변속 이상 (속도 변화가 큰 경우)
    if len(raw_df) > 1:
        raw_df['speed_diff'] = raw_df['SPEED'].diff().abs()
        rapid_change_mask = raw_df['speed_diff'] > 30  # 30km/h 이상 급변
        raw_df.loc[rapid_change_mask, 'anomaly_score'] += 0.2
        print(f"     급변속 이상 (> 30km/h 변화): {rapid_change_mask.sum():,}건")
    
    # 2. 브레이크 기반 이상탐지
    if 'BRAKE_STATUS' in raw_df.columns:
        print("   🛑 급제동 기반 이상탐지...")
        brake_mask = raw_df['BRAKE_STATUS'] == 1
        raw_df.loc[brake_mask, 'anomaly_score'] += 0.15
        print(f"     급제동 이상: {brake_mask.sum():,}건")
    
    # 3. 라벨 기반 이상탐지 (더 신중하게)
    if label_data:
        print("   🏷️ 라벨 기반 이상탐지...")
        
        # 라벨을 딕셔너리로 변환
        label_dict = {}
        for label in label_data:
            annotation = label.get('Annotation', {})
            vehicle_id = annotation.get('Vehicle_ID')
            if vehicle_id:
                if vehicle_id not in label_dict:
                    label_dict[vehicle_id] = []
                label_dict[vehicle_id].append(annotation)
        
        hazard_count = 0
        for vehicle_id, vehicle_labels in label_dict.items():
            vehicle_mask = raw_df['TRIP_ID'] == vehicle_id
            
            for annotation in vehicle_labels:
                # Hazard=True인 경우만 높은 이상점수
                if annotation.get('Hazard') == 'True':
                    raw_df.loc[vehicle_mask, 'anomaly_score'] += 0.4
                    hazard_count += vehicle_mask.sum()
        
        print(f"     위험 라벨 이상: {hazard_count:,}건")
    
    # 4. 이상 점수 정규화 (0-1 범위)
    raw_df['anomaly_score'] = np.clip(raw_df['anomaly_score'], 0, 1)
    
    # 5. 이진 라벨 생성 (임계값 0.4로 상향조정)
    raw_df['is_anomaly'] = (raw_df['anomaly_score'] >= 0.4).astype(int)
    
    # 통계 출력
    total_records = len(raw_df)
    anomaly_records = raw_df['is_anomaly'].sum()
    anomaly_ratio = anomaly_records / total_records
    
    print(f"   ✅ 이상탐지 라벨 생성 완료:")
    print(f"     전체 레코드: {total_records:,}개")
    print(f"     이상 레코드: {anomaly_records:,}개")
    print(f"     이상 비율: {anomaly_ratio*100:.2f}%")
    print(f"     평균 이상 점수: {raw_df['anomaly_score'].mean():.3f}")
    
    return raw_df

def create_grid_system(raw_df, grid_size=0.01):
    """
    🗺️ 광주 지역 그리드 시스템 생성 (최적화된 격자 크기)
    
    Args:
        raw_df: V2X 차량 데이터
        grid_size: 격자 크기 (도 단위, 0.01도 ≈ 1.1km)
    
    Returns:
        grid_centers: 격자 중심점들
        vehicle_to_grid: 차량 레코드별 소속 격자 매핑
        grid_info: 격자 정보
    """
    print(f"🗺️ 광주 지역 그리드 시스템 생성 (격자 크기: {grid_size:.3f}도 ≈ {int(grid_size * 111)}km)")
    
    # 1. 위치 데이터 확인
    bounds = GWANGJU_BOUNDS
    
    # 실제 데이터 범위 vs 광주 범위
    actual_lon_range = (raw_df['LONGITUDE'].min(), raw_df['LONGITUDE'].max())
    actual_lat_range = (raw_df['LATITUDE'].min(), raw_df['LATITUDE'].max())
    
    print(f"   📊 데이터 범위:")
    print(f"     경도: {actual_lon_range[0]:.6f} ~ {actual_lon_range[1]:.6f}")
    print(f"     위도: {actual_lat_range[0]:.6f} ~ {actual_lat_range[1]:.6f}")
    
    # 2. 격자 범위 설정 (실제 데이터 범위 + 여유공간)
    margin = grid_size * 0.5
    min_lon = max(bounds['lon_min'], actual_lon_range[0] - margin)
    max_lon = min(bounds['lon_max'], actual_lon_range[1] + margin)
    min_lat = max(bounds['lat_min'], actual_lat_range[0] - margin)
    max_lat = min(bounds['lat_max'], actual_lat_range[1] + margin)
    
    # 3. 격자 생성
    lon_grids = np.arange(min_lon, max_lon + grid_size, grid_size)
    lat_grids = np.arange(min_lat, max_lat + grid_size, grid_size)
    
    print(f"   📐 격자 정보:")
    print(f"     경도 격자: {len(lon_grids)-1}개")
    print(f"     위도 격자: {len(lat_grids)-1}개")
    print(f"     총 격자: {(len(lon_grids)-1) * (len(lat_grids)-1)}개")
    
    # 격자 수 체크 (너무 많으면 격자 크기 조정)
    total_grids = (len(lon_grids)-1) * (len(lat_grids)-1)
    if total_grids > 500:  # 500개 초과시 격자 크기 증가
        new_grid_size = grid_size * 2
        print(f"   ⚠️ 격자 수가 너무 많음 ({total_grids}개). 격자 크기를 {new_grid_size:.3f}도로 증가")
        return create_grid_system(raw_df, new_grid_size)
    
    # 4. 격자 중심점 및 ID 생성
    grid_centers = []
    grid_ids = []
    grid_mapping = {}  # (lon_idx, lat_idx) -> grid_index
    
    grid_index = 0
    for i, lon in enumerate(lon_grids[:-1]):
        for j, lat in enumerate(lat_grids[:-1]):
            center_lon = lon + grid_size / 2
            center_lat = lat + grid_size / 2
            grid_centers.append([center_lon, center_lat])
            grid_ids.append(f"Grid_{i}_{j}")
            grid_mapping[(i, j)] = grid_index
            grid_index += 1
    
    grid_centers = np.array(grid_centers)
    print(f"   ✅ 전체 격자 생성: {len(grid_centers)}개")
    
    # 5. 차량 레코드를 격자에 할당
    print("   🚗 차량 레코드를 격자에 할당 중...")
    
    vehicle_to_grid = {}
    grid_record_counts = np.zeros(len(grid_centers))
    
    for idx in raw_df.index:
        lon = raw_df.loc[idx, 'LONGITUDE']
        lat = raw_df.loc[idx, 'LATITUDE']
        
        # 해당 레코드가 속한 격자 찾기
        lon_idx = int((lon - min_lon) / grid_size)
        lat_idx = int((lat - min_lat) / grid_size)
        
        # 범위 체크
        lon_idx = max(0, min(lon_idx, len(lon_grids) - 2))
        lat_idx = max(0, min(lat_idx, len(lat_grids) - 2))
        
        if (lon_idx, lat_idx) in grid_mapping:
            grid_idx = grid_mapping[(lon_idx, lat_idx)]
            vehicle_to_grid[idx] = grid_idx
            grid_record_counts[grid_idx] += 1
    
    print(f"   📍 레코드 할당 완료: {len(vehicle_to_grid):,}개 레코드")
    
    # 6. 활성 격자 선별 (레코드가 있는 격자만)
    active_mask = grid_record_counts > 0
    active_centers = grid_centers[active_mask]
    active_ids = [grid_ids[i] for i in range(len(grid_ids)) if active_mask[i]]
    active_counts = grid_record_counts[active_mask]
    
    # 차량 매핑 업데이트 (활성 격자 인덱스로)
    old_to_new_mapping = {}
    new_idx = 0
    for old_idx in range(len(grid_centers)):
        if active_mask[old_idx]:
            old_to_new_mapping[old_idx] = new_idx
            new_idx += 1
    
    updated_vehicle_to_grid = {}
    for record_idx, old_grid_idx in vehicle_to_grid.items():
        if old_grid_idx in old_to_new_mapping:
            updated_vehicle_to_grid[record_idx] = old_to_new_mapping[old_grid_idx]
    
    print(f"   🔥 활성 격자 (데이터 있음): {len(active_centers)}개")
    print(f"   📊 격자별 평균 레코드 수: {active_counts.mean():.1f}개")
    print(f"   📊 격자별 레코드 수 범위: {active_counts.min():.0f} ~ {active_counts.max():.0f}개")
    
    # 격자 정보 반환
    grid_info = {
        'grid_size': grid_size,
        'total_grids': len(grid_centers),
        'active_grids': len(active_centers),
        'grid_centers': active_centers,
        'grid_ids': active_ids,
        'grid_counts': active_counts,
        'lon_range': (min_lon, max_lon),
        'lat_range': (min_lat, max_lat)
    }
    
    return active_centers, updated_vehicle_to_grid, grid_info

def create_grid_adjacency_matrix(grid_centers, connection_threshold=1500):
    """🔗 격자 간 인접행렬 생성"""
    print(f"🔗 격자 인접행렬 생성 (연결 임계값: {connection_threshold}m)")
    
    num_grids = len(grid_centers)
    adjacency_matrix = np.zeros((num_grids, num_grids))
    
    for i in range(num_grids):
        for j in range(num_grids):
            if i == j:
                adjacency_matrix[i, j] = 1.0  # 자기 자신과는 완전 연결
            else:
                # 격자 간 거리 계산 (미터 단위)
                dist_lon = (grid_centers[i, 0] - grid_centers[j, 0]) * 111000  # 1도 ≈ 111km
                dist_lat = (grid_centers[i, 1] - grid_centers[j, 1]) * 111000
                distance = np.sqrt(dist_lon**2 + dist_lat**2)
                
                # 연결 강도 계산 (거리 기반 지수 감소)
                if distance < connection_threshold:
                    connection_strength = np.exp(-distance / connection_threshold)
                    adjacency_matrix[i, j] = connection_strength
                else:
                    adjacency_matrix[i, j] = 0.0
    
    # 연결 통계
    connection_ratio = np.count_nonzero(adjacency_matrix) / (num_grids * num_grids)
    avg_connections = np.count_nonzero(adjacency_matrix, axis=1).mean()
    
    print(f"   ✅ 격자 인접행렬 완성: {num_grids}×{num_grids}")
    print(f"   📊 연결 비율: {connection_ratio:.3f}")
    print(f"   📊 격자당 평균 연결 수: {avg_connections:.1f}개")
    
    return adjacency_matrix

def create_grid_anomaly_matrix(raw_df, vehicle_to_grid, num_grids, time_interval='15min'):
    """🚨 격자 기반 시간×격자 이상점수 행렬 생성"""
    print(f"🚨 격자 기반 이상점수 행렬 생성 (간격: {time_interval})")
    
    # 1. 시간 데이터 전처리
    if 'ISSUE_DATE' in raw_df.columns:
        try:
            # ISSUE_DATE 파싱 시도 (다양한 형식 시도)
            raw_df['datetime'] = pd.to_datetime(
                raw_df['ISSUE_DATE'].astype(str), 
                format='%Y%m%d%H%M%S', 
                errors='coerce'
            )
            
            # 파싱 실패한 경우 다른 형식 시도
            if raw_df['datetime'].isna().sum() > len(raw_df) * 0.5:
                raw_df['datetime'] = pd.to_datetime(
                    raw_df['ISSUE_DATE'], 
                    errors='coerce'
                )
            
            # 여전히 실패하면 균등 분포로 생성
            if raw_df['datetime'].isna().sum() > len(raw_df) * 0.5:
                raise ValueError("Too many parsing failures")
            
            print(f"   ✅ 시간 파싱 성공: {raw_df['datetime'].min()} ~ {raw_df['datetime'].max()}")
            
        except:
            print(f"   ⚠️ ISSUE_DATE 파싱 실패, 8월 내 균등 분포로 생성")
            # 8월 첫 5일 내 균등 분포로 생성
            start_date = pd.Timestamp('2022-08-01')
            end_date = pd.Timestamp('2022-08-05 23:59:59')
            raw_df['datetime'] = pd.date_range(start_date, end_date, periods=len(raw_df))
    else:
        print(f"   ⚠️ ISSUE_DATE 컬럼 없음, 8월 내 균등 분포로 생성")
        start_date = pd.Timestamp('2022-08-01')
        end_date = pd.Timestamp('2022-08-05 23:59:59')
        raw_df['datetime'] = pd.date_range(start_date, end_date, periods=len(raw_df))
    
    # 2. 시간 간격별 그룹화
    raw_df['time_bin'] = raw_df['datetime'].dt.floor(time_interval)
    
    # 3. 격자 정보 추가 (레코드별)
    raw_df['grid_id'] = raw_df.index.map(vehicle_to_grid)
    
    # 4. 격자 매핑이 없는 레코드 제거
    before_filter = len(raw_df)
    raw_df = raw_df.dropna(subset=['grid_id'])
    raw_df['grid_id'] = raw_df['grid_id'].astype(int)
    after_filter = len(raw_df)
    
    print(f"   📍 격자 매핑된 레코드: {after_filter:,}개 ({after_filter/before_filter*100:.1f}%)")
    
    # 5. 격자별 시간대별 이상점수 집계
    print("   📊 격자별 시간대별 이상점수 집계 중...")
    
    # 그룹별 평균 이상점수 계산
    grid_time_anomaly = raw_df.groupby(['time_bin', 'grid_id'])['anomaly_score'].agg([
        'mean',  # 평균 이상점수
        'max',   # 최대 이상점수  
        'count'  # 레코드 수
    ]).reset_index()
    
    # 피벗 테이블 생성 (시간 × 격자)
    anomaly_pivot = grid_time_anomaly.pivot_table(
        index='time_bin', 
        columns='grid_id', 
        values='mean',  # 평균 이상점수 사용
        fill_value=0.0
    )
    
    # 6. 모든 격자가 포함되도록 보정
    all_grid_ids = list(range(num_grids))
    for grid_id in all_grid_ids:
        if grid_id not in anomaly_pivot.columns:
            anomaly_pivot[grid_id] = 0.0
    
    # 격자 순서대로 정렬
    anomaly_pivot = anomaly_pivot.reindex(columns=all_grid_ids, fill_value=0.0)
    
    print(f"   ✅ 격자 이상점수 행렬 완성: {anomaly_pivot.shape} (시간 × 격자)")
    print(f"   📊 시간 범위: {anomaly_pivot.index.min()} ~ {anomaly_pivot.index.max()}")
    print(f"   🚨 전체 평균 이상점수: {anomaly_pivot.values.mean():.3f}")
    
    return anomaly_pivot.values, list(anomaly_pivot.index)

def create_grid_poi_features(grid_centers, grid_info):
    """🏢 격자별 정적 속성 생성 (광주 특화)"""
    print("🏢 격자별 정적 속성 (POI) 생성 - 광주 특화")
    
    grid_features = []
    num_grids = len(grid_centers)
    
    # 광주 지역 중심점들
    gwangju_city_center = (126.9, 35.15)  # 광주 시청 근처
    buk_gu_center = (126.92, 35.18)       # 북구 중심
    dong_gu_center = (126.92, 35.14)      # 동구 중심
    seo_gu_center = (126.88, 35.15)       # 서구 중심
    nam_gu_center = (126.90, 35.12)       # 남구 중심
    gwangsan_gu_center = (126.95, 35.20)  # 광산구 중심
    
    for i, (center_lon, center_lat) in enumerate(grid_centers):
        features = []
        
        # 1. 기본 위치 특성
        features.append(center_lon)  # 경도
        features.append(center_lat)  # 위도
        
        # 2. 광주 시청(도심)과의 거리
        dist_to_center = np.sqrt(
            ((center_lon - gwangju_city_center[0]) * 111000) ** 2 +
            ((center_lat - gwangju_city_center[1]) * 111000) ** 2
        )
        features.append(dist_to_center / 1000)  # km 단위
        
        # 3. 도심/외곽 구분
        is_downtown = 1.0 if dist_to_center < 2000 else 0.0  # 2km 이내는 도심
        features.append(is_downtown)
        
        # 4. 구별 특성 (가장 가까운 구 중심)
        districts = {
            'buk': buk_gu_center,
            'dong': dong_gu_center,
            'seo': seo_gu_center,
            'nam': nam_gu_center,
            'gwangsan': gwangsan_gu_center
        }
        
        district_distances = {}
        for district, (d_lon, d_lat) in districts.items():
            dist = np.sqrt(
                ((center_lon - d_lon) * 111000) ** 2 +
                ((center_lat - d_lat) * 111000) ** 2
            )
            district_distances[district] = dist
        
        # 가장 가까운 구
        closest_district = min(district_distances, key=district_distances.get)
        
        # 구별 원핫 인코딩
        for district in districts.keys():
            features.append(1.0 if district == closest_district else 0.0)
        
        # 5. 교통 특성
        # 시내 중심에서 가까울수록 교통 밀도 높음
        traffic_density = max(0.1, 1.0 - dist_to_center / 5000)
        features.append(traffic_density)
        
        # 6. 지역 특성 시뮬레이션
        # 상업지역 점수 (도심 + 동구에서 높음)
        commercial_score = 0.8 if is_downtown else 0.3
        if closest_district == 'dong':  # 동구는 상업지역
            commercial_score = max(commercial_score, 0.6)
        features.append(commercial_score)
        
        # 주거지역 점수 (광산구, 북구에서 높음)
        residential_score = 0.7 if closest_district in ['gwangsan', 'buk'] else 0.4
        features.append(residential_score)
        
        # 산업지역 점수 (광산구에서 높음)
        industrial_score = 0.8 if closest_district == 'gwangsan' else 0.2
        features.append(industrial_score)
        
        # 7. 레코드 밀도
        if i < len(grid_info['grid_counts']):
            record_density = min(1.0, grid_info['grid_counts'][i] / 1000)
        else:
            record_density = 0.1
        features.append(record_density)
        
        grid_features.append(features)
    
    grid_matrix = np.array(grid_features)
    
    print(f"   ✅ 격자 정적 속성 완성: {grid_matrix.shape} (격자 × 특성)")
    feature_names = ['경도', '위도', '시청거리', '도심여부', '북구', '동구', '서구', '남구', '광산구', 
                    '교통밀도', '상업점수', '주거점수', '산업점수', '레코드밀도']
    print(f"   📊 특성 목록: {feature_names}")
    
    return grid_matrix

def create_time_weather_features(time_index):
    """🌤️ 시간별 동적 속성 생성"""
    print("🌤️ 시간별 동적 속성 (Weather) 생성")
    
    weather_features = []
    
    for timestamp in time_index:
        features = []
        
        # 1. 시간 패턴
        hour = timestamp.hour
        
        # Period 원핫 인코딩
        period_f = 1.0 if 6 <= hour < 12 else 0.0   # 오전
        period_a = 1.0 if 12 <= hour < 18 else 0.0  # 오후
        period_n = 1.0 if 18 <= hour < 24 else 0.0  # 밤
        period_d = 1.0 if 0 <= hour < 6 else 0.0    # 새벽
        
        features.extend([period_f, period_a, period_n, period_d])
        
        # 2. 요일 정보
        weekday = timestamp.weekday()
        is_weekend = 1.0 if weekday >= 5 else 0.0
        is_weekday = 1.0 - is_weekend
        
        features.extend([is_weekday, is_weekend])
        
        # 3. 교통 패턴
        rush_morning = 1.0 if 7 <= hour <= 9 else 0.0
        rush_evening = 1.0 if 17 <= hour <= 19 else 0.0
        lunch_time = 1.0 if 11 <= hour <= 13 else 0.0
        night_time = 1.0 if 22 <= hour <= 6 else 0.0
        
        features.extend([rush_morning, rush_evening, lunch_time, night_time])
        
        # 4. 정규화된 시간 특성
        features.append(hour / 24.0)
        features.append(weekday / 6.0)
        features.append(timestamp.day / 31.0)
        
        # 5. 이상 위험도 (광주 교통 패턴 반영)
        anomaly_risk = 0.1
        if rush_morning or rush_evening:
            anomaly_risk = 0.3 + np.random.normal(0, 0.05)
        elif night_time:
            anomaly_risk = 0.25 + np.random.normal(0, 0.03)
        else:
            anomaly_risk = 0.1 + np.random.normal(0, 0.02)
        
        features.append(np.clip(anomaly_risk, 0, 1))
        
        # 6. 날씨 시뮬레이션 (8월 여름)
        weather_impact = 0.6 + np.random.normal(0, 0.1)  # 여름철 높은 온도
        features.append(np.clip(weather_impact, 0, 1))
        
        weather_features.append(features)
    
    weather_matrix = np.array(weather_features)
    
    print(f"   ✅ 동적 속성 완성: {weather_matrix.shape} (시간 × 특성)")
    
    return weather_matrix

def save_grid_astgcn_format(anomaly_matrix, grid_adjacency, grid_poi, weather_matrix, 
                           grid_info, time_index, output_dir='v2x_astgcn_data'):
    """💾 광주 지역 그리드 기반 AST-GCN 형식으로 데이터 저장"""
    print(f"💾 광주 지역 그리드 기반 AST-GCN 데이터 저장: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 이상점수 행렬 저장
    anomaly_df = pd.DataFrame(anomaly_matrix)
    anomaly_path = os.path.join(output_dir, 'v2x_speed.csv')
    anomaly_df.to_csv(anomaly_path, header=False, index=False)
    print(f"   ✅ {anomaly_path}: {anomaly_matrix.shape}")
    
    # 2. 격자 인접행렬 저장
    adj_df = pd.DataFrame(grid_adjacency)
    adj_path = os.path.join(output_dir, 'v2x_adj.csv')
    adj_df.to_csv(adj_path, header=False, index=False)
    print(f"   ✅ {adj_path}: {grid_adjacency.shape}")
    
    # 3. 격자별 정적 속성 저장
    poi_df = pd.DataFrame(grid_poi)
    poi_path = os.path.join(output_dir, 'v2x_poi.csv')
    poi_df.to_csv(poi_path, header=False, index=False)
    print(f"   ✅ {poi_path}: {grid_poi.shape}")
    
    # 4. 시간별 동적 속성 저장
    weather_df = pd.DataFrame(weather_matrix)
    weather_path = os.path.join(output_dir, 'v2x_weather.csv')
    weather_df.to_csv(weather_path, header=False, index=False)
    print(f"   ✅ {weather_path}: {weather_matrix.shape}")
    
    # 5. 메타데이터 저장
    metadata = {
        'task_type': 'anomaly_detection',
        'region': 'gwangju',
        'node_type': 'regional_grid',
        'grid_info': {
            'grid_size_degrees': grid_info['grid_size'],
            'grid_size_meters': int(grid_info['grid_size'] * 111000),
            'active_grids': grid_info['active_grids'],
            'region_bounds': GWANGJU_BOUNDS
        },
        'data_dimensions': {
            'time_steps': anomaly_matrix.shape[0],
            'num_grids': anomaly_matrix.shape[1],
            'poi_features': grid_poi.shape[1],
            'weather_features': weather_matrix.shape[1]
        },
        'time_range': {
            'start': str(time_index[0]),
            'end': str(time_index[-1]),
            'interval': '15min'
        },
        'average_anomaly_score': float(anomaly_matrix.mean())
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {metadata_path}: 광주 지역 메타데이터")

def convert_v2x_to_gwangju_grid_anomaly_detection(data_dir='data/daily_merged/08월', 
                                                 output_dir='v2x_astgcn_data',
                                                 grid_size=0.01,
                                                 time_interval='15min'):
    """🎯 광주 V2X 데이터를 지역 그리드 기반 이상탐지 형식으로 변환"""
    print("=" * 80)
    print("🗺️ 광주 V2X → 지역 그리드 기반 이상탐지 변환 시작")
    print("=" * 80)
    
    # 1. V2X 원본 데이터 로딩 (처음 5일치만)
    raw_df, label_data = load_v2x_data(data_dir)
    
    # 2. 위치 데이터 정제 및 광주 범위 필터링
    raw_df = clean_and_filter_location_data(raw_df)
    
    # 3. 이상상황 라벨 생성
    raw_df = create_anomaly_labels(raw_df, label_data)
    
    # 4. 광주 지역 그리드 시스템 생성
    grid_centers, vehicle_to_grid, grid_info = create_grid_system(raw_df, grid_size)
    
    # 5. 격자 간 인접행렬 생성
    grid_adjacency = create_grid_adjacency_matrix(grid_centers)
    
    # 6. 격자 기반 이상점수 행렬 생성
    anomaly_matrix, time_index = create_grid_anomaly_matrix(
        raw_df, vehicle_to_grid, len(grid_centers), time_interval
    )
    
    # 7. 격자별 정적 속성 생성 (광주 특화)
    grid_poi = create_grid_poi_features(grid_centers, grid_info)
    
    # 8. 시간별 동적 속성 생성
    weather_matrix = create_time_weather_features(time_index)
    
    # 9. AST-GCN 형식으로 저장
    save_grid_astgcn_format(anomaly_matrix, grid_adjacency, grid_poi, weather_matrix,
                            grid_info, time_index, output_dir)
    
    print("=" * 80)
    print("✅ 광주 지역 그리드 기반 V2X 이상탐지 변환 완료!")
    print(f"📂 출력 위치: {output_dir}")
    print(f"🗺️ 활성 격자 수: {len(grid_centers)}")
    print(f"⏰ 시간 스텝: {len(time_index)}")
    print(f"🚨 평균 이상점수: {anomaly_matrix.mean():.3f}")
    print("=" * 80)
    
    return anomaly_matrix, grid_adjacency, grid_poi, weather_matrix, grid_centers

# 메인 실행 부분
if __name__ == "__main__":
    # 설정 (광주 특화)
    DATA_DIR = "data/daily_merged/08월"
    OUTPUT_DIR = "v2x_astgcn_data"
    GRID_SIZE = 0.01  # 1.1km 격자 (관리 가능한 크기)
    TIME_INTERVAL = "15min"  # 30분 간격
    
    # 데이터 확인
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {DATA_DIR}")
        exit(1)
    
    try:
        print("🚀 광주 지역 그리드 기반 V2X 이상탐지 변환 시작!")
        print(f"   📊 설정:")
        print(f"     데이터 경로: {DATA_DIR}")
        print(f"     출력 경로: {OUTPUT_DIR}")
        print(f"     격자 크기: {GRID_SIZE}도 (약 {int(GRID_SIZE * 111)}km)")
        print(f"     시간 간격: {TIME_INTERVAL}")
        print(f"     대상 지역: 광주광역시")
        
        # 변환 실행
        anomaly_matrix, grid_adjacency, grid_poi, weather_matrix, grid_centers = convert_v2x_to_gwangju_grid_anomaly_detection(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            grid_size=GRID_SIZE,
            time_interval=TIME_INTERVAL
        )
        
        print("\n🎉 광주 지역 그리드 변환 성공!")
        print(f"\n📊 최종 데이터 요약:")
        print(f"   🗺️ 격자 수: {len(grid_centers)}개")
        print(f"   ⏰ 시간 스텝: {anomaly_matrix.shape[0]}개")
        print(f"   🚨 평균 이상점수: {anomaly_matrix.mean():.3f}")
        print(f"   📈 이상 비율: {(anomaly_matrix > 0.3).mean()*100:.2f}%")
        
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()