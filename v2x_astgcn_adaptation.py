#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2X 데이터를 AST-GCN 논문 형식으로 변환하는 스크립트

입력:
- data/daily_merged/8월/220801_C_raw.csv (V2X 주행 데이터)
- data/daily_merged/8월/220801_C_label.jsonl (V2X 라벨 데이터)

출력:
- v2x_astgcn_data/v2x_speed.csv     (시간×차량 속도행렬)
- v2x_astgcn_data/v2x_adj.csv       (차량×차량 인접행렬)
- v2x_astgcn_data/v2x_poi.csv       (차량별 정적 속성)
- v2x_astgcn_data/v2x_weather.csv   (시간별 동적 속성)

Author: Claude (Anthropic)
Date: 2025-06-03
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_v2x_data(data_dir):
    """
    V2X 원본 데이터 로딩 (모든 파일)
    
    Args:
        data_dir (str): V2X 데이터 폴더 경로 (예: "data/daily_merged/8월")
    
    Returns:
        raw_df (DataFrame): 모든 주행 데이터 합쳐진 것
        label_data (list): 모든 라벨 데이터 합쳐진 것
    """
    print(f"📂 V2X 데이터 로딩: {data_dir}")
    
    # 1. 모든 주행 데이터 로딩
    raw_files = [f for f in os.listdir(data_dir) if f.endswith('_raw.csv')]
    if not raw_files:
        raise FileNotFoundError(f"❌ {data_dir}에서 *_raw.csv 파일을 찾을 수 없습니다")
    
    print(f"   📄 발견된 주행 데이터 파일: {len(raw_files)}개")
    
    # 모든 raw 파일을 합치기
    all_raw_data = []
    for raw_file in raw_files:
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
    
    # 2. 모든 라벨 데이터 로딩
    label_files = [f for f in os.listdir(data_dir) if f.endswith('_label.jsonl')]
    label_data = []
    
    if label_files:
        print(f"   📄 발견된 라벨 데이터 파일: {len(label_files)}개")
        
        for label_file in label_files:
            file_path = os.path.join(data_dir, label_file)
            try:
                with open(file_path, 'r') as f:
                    file_labels = []
                    for line in f:
                        file_labels.append(json.loads(line.strip()))
                    label_data.extend(file_labels)
                    print(f"       ✅ {label_file}: {len(file_labels)}개")
            except Exception as e:
                print(f"       ❌ {label_file} 로딩 실패: {e}")
        
        print(f"   📊 통합 라벨 데이터: {len(label_data)}개")
    else:
        print("   ⚠️ 라벨 데이터가 없습니다. 기본값으로 진행합니다.")
    
    return raw_df, label_data

def create_time_vehicle_matrix(raw_df, time_interval='15min'):
    """
    V2X 데이터를 시간×차량 속도 행렬로 변환
    """
    print(f"⏰ 시간×차량 행렬 생성 (간격: {time_interval})")
    
    # 필수 컬럼 확인 및 매핑
    print(f"   🔍 원본 컬럼들: {raw_df.columns.tolist()}")
    
    # 1. TRIP_ID 매핑
    if 'TRIP_ID' not in raw_df.columns:
        if 'VEHICLE_ID' in raw_df.columns:
            raw_df['TRIP_ID'] = raw_df['VEHICLE_ID']
            print(f"   ✅ VEHICLE_ID → TRIP_ID 매핑")
        else:
            print(f"   ❌ 차량 ID 컬럼을 찾을 수 없습니다")
    
    # 2. SPEED 확인
    if 'SPEED' not in raw_df.columns:
        for alt_col in ['speed', 'Speed', 'velocity', 'VELOCITY']:
            if alt_col in raw_df.columns:
                raw_df['SPEED'] = raw_df[alt_col]
                print(f"   ✅ {alt_col} → SPEED 매핑")
                break
        else:
            print(f"   ❌ 속도 컬럼을 찾을 수 없습니다")
    
    # 3. 시간 컬럼 확인 (이 부분이 핵심!)
    time_column = None
    if 'ISSUE_DATE' in raw_df.columns:
        time_column = 'ISSUE_DATE'
        print(f"   ✅ 시간 컬럼 발견: ISSUE_DATE")
    elif 'TIMESTAMP' in raw_df.columns:
        time_column = 'TIMESTAMP'
        print(f"   ✅ 시간 컬럼 발견: TIMESTAMP")
    else:
        for alt_col in ['timestamp', 'time', 'TIME', 'datetime']:
            if alt_col in raw_df.columns:
                time_column = alt_col
                print(f"   ✅ 시간 컬럼 발견: {alt_col}")
                break
        else:
            print(f"   ❌ 시간 컬럼을 찾을 수 없습니다")

    # 시간 파싱
    if time_column:
        print(f"   🔍 시간 컬럼: {time_column}")
        print(f"   🔍 {time_column} 타입: {raw_df[time_column].dtype}")
        print(f"   🔍 {time_column} 샘플: {raw_df[time_column].head().tolist()}")
        
        try:
            # ISSUE_DATE 형식: 20220808170730 (YYYYMMDDHHMMSS)
            if time_column == 'ISSUE_DATE':
                # int64를 문자열로 변환하여 파싱
                if raw_df[time_column].dtype in ['int64', 'float64']:
                    raw_df['datetime'] = pd.to_datetime(raw_df[time_column].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
                else:
                    raw_df['datetime'] = pd.to_datetime(raw_df[time_column], format='%Y%m%d%H%M%S', errors='coerce')
            else:
                # 다른 시간 형식들
                raw_df['datetime'] = pd.to_datetime(raw_df[time_column], errors='coerce')
            
            # 파싱 실패한 행들 확인
            failed_parsing = raw_df['datetime'].isna().sum()
            if failed_parsing > 0:
                print(f"   ⚠️ 파싱 실패한 행: {failed_parsing}개 (전체의 {failed_parsing/len(raw_df)*100:.2f}%)")
                # 실패한 행들은 제거
                raw_df = raw_df.dropna(subset=['datetime'])
            
            if len(raw_df) > 0:
                print(f"   ✅ {time_column} 파싱 성공: {raw_df['datetime'].min()} ~ {raw_df['datetime'].max()}")
                print(f"   📊 유효한 데이터: {len(raw_df)}행")
            else:
                raise ValueError(f"모든 {time_column} 파싱이 실패했습니다.")
                
        except Exception as e:
            print(f"   ❌ {time_column} 파싱 실패: {e}")
            print("   → 8월 내 시간으로 분산 생성")
            start_date = pd.Timestamp('2022-08-01')
            end_date = pd.Timestamp('2022-08-31 23:59:59')
            raw_df['datetime'] = pd.date_range(start_date, end_date, periods=len(raw_df))
    else:
        print("   ⚠️ 시간 컬럼 없음, 8월 내 시간으로 분산 생성")
        start_date = pd.Timestamp('2022-08-01')
        end_date = pd.Timestamp('2022-08-31 23:59:59')
        raw_df['datetime'] = pd.date_range(start_date, end_date, periods=len(raw_df))
    
    # 시간 간격별 그룹화
    raw_df['time_bin'] = raw_df['datetime'].dt.floor(time_interval)
    
    # 차량별 시간대별 평균 속도 계산
    speed_pivot = raw_df.groupby(['time_bin', 'TRIP_ID'])['SPEED'].mean().unstack(fill_value=0)
    
    print(f"   ✅ 행렬 크기: {speed_pivot.shape} (시간 × 차량)")
    print(f"   📊 시간 범위: {speed_pivot.index.min()} ~ {speed_pivot.index.max()}")
    print(f"   🚗 차량 수: {speed_pivot.shape[1]}개")
    
    return speed_pivot.values, list(speed_pivot.columns), list(speed_pivot.index)

def create_adjacency_matrix(raw_df, vehicle_ids, distance_threshold=500):
    """
    차량 간 인접행렬 생성 (거리 기반)
    
    Args:
        raw_df (DataFrame): V2X 주행 데이터
        vehicle_ids (list): 차량 ID 목록
        distance_threshold (float): 인접 기준 거리 (미터)
    
    Returns:
        adj_matrix (ndarray): [num_vehicles, num_vehicles] 인접행렬
    """
    print(f"🗺️ 인접행렬 생성 (거리 임계값: {distance_threshold}m)")
    
    num_vehicles = len(vehicle_ids)
    adj_matrix = np.zeros((num_vehicles, num_vehicles))
    
    # 차량별 평균 위치 계산
    if 'LONGITUDE' in raw_df.columns and 'LATITUDE' in raw_df.columns:
        vehicle_positions = {}
        
        for i, vid in enumerate(vehicle_ids):
            vehicle_data = raw_df[raw_df['TRIP_ID'] == vid]
            if len(vehicle_data) > 0:
                avg_lon = vehicle_data['LONGITUDE'].mean()
                avg_lat = vehicle_data['LATITUDE'].mean()
                vehicle_positions[i] = (avg_lon, avg_lat)
        
        # 차량 간 거리 계산 및 인접행렬 구성
        for i in range(num_vehicles):
            for j in range(num_vehicles):
                if i != j and i in vehicle_positions and j in vehicle_positions:
                    # 단순 유클리드 거리 (실제로는 지구 거리 계산 필요)
                    pos_i = vehicle_positions[i]
                    pos_j = vehicle_positions[j]
                    distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2) * 111000  # 대략적 거리 변환
                    
                    if distance < distance_threshold:
                        adj_matrix[i, j] = np.exp(-distance / distance_threshold)  # 가중치
        
        # 자기 자신과의 연결
        np.fill_diagonal(adj_matrix, 1.0)
        
        connection_ratio = np.count_nonzero(adj_matrix) / (num_vehicles * num_vehicles)
        print(f"   ✅ 인접행렬 완성: {num_vehicles}×{num_vehicles}")
        print(f"   📊 연결 비율: {connection_ratio:.3f}")
        
    else:
        print("   ⚠️ 위치 정보 없음, 완전연결 그래프로 생성")
        adj_matrix = np.ones((num_vehicles, num_vehicles))
    
    return adj_matrix

def create_poi_features(raw_df, label_data, vehicle_ids):
    """
    차량별 정적 속성 (POI 역할) 생성
    
    Args:
        raw_df (DataFrame): V2X 주행 데이터
        label_data (list): V2X 라벨 데이터
        vehicle_ids (list): 차량 ID 목록
    
    Returns:
        poi_matrix (ndarray): [num_vehicles, num_features] 정적 속성 행렬
    """
    print("🏢 차량별 정적 속성 (POI) 생성")
    
    num_vehicles = len(vehicle_ids)
    poi_features = []
    
    # 라벨 데이터를 딕셔너리로 변환
    label_dict = {}
    for label in label_data:
        if 'TRIP_ID' in label:
            label_dict[label['TRIP_ID']] = label
    
    for i, vid in enumerate(vehicle_ids):
        vehicle_data = raw_df[raw_df['TRIP_ID'] == vid]
        label_info = label_dict.get(vid, {})
        
        features = []
        
        # 1. 주행 패턴 특성
        if len(vehicle_data) > 0:
            features.append(vehicle_data['SPEED'].mean())           # 평균 속도
            features.append(vehicle_data['SPEED'].std())            # 속도 변동성
            features.append(vehicle_data['SPEED'].max())            # 최대 속도
            features.append(vehicle_data['SPEED'].min())            # 최소 속도
        else:
            features.extend([30.0, 10.0, 60.0, 0.0])  # 기본값
        
        # 2. 위치 특성
        if 'LONGITUDE' in vehicle_data.columns and len(vehicle_data) > 0:
            features.append(vehicle_data['LONGITUDE'].mean())       # 평균 경도
            features.append(vehicle_data['LATITUDE'].mean())        # 평균 위도
            features.append(vehicle_data['LONGITUDE'].std())        # 위치 변동성 (경도)
            features.append(vehicle_data['LATITUDE'].std())         # 위치 변동성 (위도)
        else:
            features.extend([127.0, 37.0, 0.01, 0.01])  # 서울 기본값
        
        # 3. 라벨 기반 행동 특성
        turn_pref = 0.5  # 기본값
        if 'Turn' in label_info:
            if label_info['Turn'] == 'Right':
                turn_pref = 1.0
            elif label_info['Turn'] == 'Left':
                turn_pref = 0.0
        features.append(turn_pref)  # 회전 선호도
        
        lane_pref = 0.5  # 기본값
        if 'Lane' in label_info:
            if label_info['Lane'] == 'R-Side':
                lane_pref = 1.0
            elif label_info['Lane'] == 'L-Side':
                lane_pref = 0.0
        features.append(lane_pref)  # 차선 선호도
        
        speed_violation = 0.0
        if 'Speed' in label_info:
            speed_violation = 1.0 if label_info['Speed'] == 'True' else 0.0
        features.append(speed_violation)  # 속도 위반 이력
        
        hazard_exp = 0.0
        if 'Hazard' in label_info:
            hazard_exp = 1.0 if label_info['Hazard'] == 'True' else 0.0
        features.append(hazard_exp)  # 위험 상황 경험
        
        poi_features.append(features)
    
    poi_matrix = np.array(poi_features)
    print(f"   ✅ 정적 속성 완성: {poi_matrix.shape} (차량 × 특성)")
    print(f"   📊 특성 목록: [평균속도, 속도변동, 최대속도, 최소속도, 평균경도, 평균위도, 경도변동, 위도변동, 회전선호, 차선선호, 속도위반, 위험경험]")
    
    return poi_matrix

def create_weather_features(time_index):
    """
    시간별 동적 속성 (Weather 역할) 생성
    
    Args:
        time_index (list): 시간 인덱스 목록
    
    Returns:
        weather_matrix (ndarray): [time_steps, num_features] 동적 속성 행렬
    """
    print("🌤️ 시간별 동적 속성 (Weather) 생성")
    
    weather_features = []
    
    for timestamp in time_index:
        features = []
        
        # 1. 시간 패턴 (Period 정보)
        hour = timestamp.hour
        
        # Period 원핫 인코딩 (F/A/N/D)
        period_f = 1.0 if 6 <= hour < 12 else 0.0   # 오전 (06-12)
        period_a = 1.0 if 12 <= hour < 18 else 0.0  # 오후 (12-18)
        period_n = 1.0 if 18 <= hour < 24 else 0.0  # 밤 (18-24)
        period_d = 1.0 if 0 <= hour < 6 else 0.0    # 새벽 (00-06)
        
        features.extend([period_f, period_a, period_n, period_d])
        
        # 2. 요일 정보
        weekday = timestamp.weekday()  # 0=월요일, 6=일요일
        is_weekend = 1.0 if weekday >= 5 else 0.0
        is_weekday = 1.0 - is_weekend
        
        features.extend([is_weekday, is_weekend])
        
        # 3. 시간대별 교통 패턴
        rush_morning = 1.0 if 7 <= hour <= 9 else 0.0    # 출근 시간
        rush_evening = 1.0 if 17 <= hour <= 19 else 0.0  # 퇴근 시간
        lunch_time = 1.0 if 11 <= hour <= 13 else 0.0    # 점심 시간
        
        features.extend([rush_morning, rush_evening, lunch_time])
        
        # 4. 추가 시간 특성
        features.append(hour / 24.0)           # 시간 (정규화)
        features.append(weekday / 6.0)         # 요일 (정규화)
        features.append(timestamp.day / 31.0)  # 날짜 (정규화)
        
        # 5. 시뮬레이션된 교통 지표 (실제 환경에서는 실시간 데이터 사용)
        # 출퇴근 시간에 혼잡도 증가
        traffic_density = 0.5
        if rush_morning or rush_evening:
            traffic_density = 0.8 + np.random.normal(0, 0.1)
        elif lunch_time:
            traffic_density = 0.6 + np.random.normal(0, 0.05)
        else:
            traffic_density = 0.3 + np.random.normal(0, 0.05)
        
        features.append(np.clip(traffic_density, 0, 1))
        
        # 6. 네트워크 상태 (V2X 특화)
        network_quality = 0.8 + np.random.normal(0, 0.1)  # 통신 품질
        features.append(np.clip(network_quality, 0, 1))
        
        weather_features.append(features)
    
    weather_matrix = np.array(weather_features)
    print(f"   ✅ 동적 속성 완성: {weather_matrix.shape} (시간 × 특성)")
    print(f"   📊 특성 목록: [Period_F, Period_A, Period_N, Period_D, 평일, 주말, 출근시간, 퇴근시간, 점심시간, 시간정규화, 요일정규화, 날짜정규화, 교통밀도, 네트워크품질]")
    
    return weather_matrix

def save_astgcn_format(speed_matrix, adj_matrix, poi_matrix, weather_matrix, 
                       vehicle_ids, time_index, output_dir='v2x_astgcn_data'):
    """
    AST-GCN 형식으로 데이터 저장
    
    Args:
        speed_matrix (ndarray): 속도 행렬
        adj_matrix (ndarray): 인접행렬
        poi_matrix (ndarray): 정적 속성 행렬
        weather_matrix (ndarray): 동적 속성 행렬
        vehicle_ids (list): 차량 ID 목록
        time_index (list): 시간 인덱스 목록
        output_dir (str): 출력 디렉토리
    """
    print(f"💾 AST-GCN 형식 저장: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 속도 행렬 저장 (CSV, header 없음)
    speed_df = pd.DataFrame(speed_matrix)
    speed_path = os.path.join(output_dir, 'v2x_speed.csv')
    speed_df.to_csv(speed_path, header=False, index=False)
    print(f"   ✅ {speed_path}: {speed_matrix.shape}")
    
    # 2. 인접행렬 저장 (CSV, header 없음)
    adj_df = pd.DataFrame(adj_matrix)
    adj_path = os.path.join(output_dir, 'v2x_adj.csv')
    adj_df.to_csv(adj_path, header=False, index=False)
    print(f"   ✅ {adj_path}: {adj_matrix.shape}")
    
    # 3. 정적 속성 저장 (CSV, header 없음)
    poi_df = pd.DataFrame(poi_matrix)
    poi_path = os.path.join(output_dir, 'v2x_poi.csv')
    poi_df.to_csv(poi_path, header=False, index=False)
    print(f"   ✅ {poi_path}: {poi_matrix.shape}")
    
    # 4. 동적 속성 저장 (CSV, header 없음)
    weather_df = pd.DataFrame(weather_matrix)
    weather_path = os.path.join(output_dir, 'v2x_weather.csv')
    weather_df.to_csv(weather_path, header=False, index=False)
    print(f"   ✅ {weather_path}: {weather_matrix.shape}")
    
    # 5. 메타데이터 저장 (참고용)
    metadata = {
        'num_vehicles': len(vehicle_ids),
        'num_time_steps': len(time_index),
        'time_range': {
            'start': str(time_index[0]),
            'end': str(time_index[-1])
        },
        'vehicle_ids': vehicle_ids[:100],  # 처음 100개만 저장 (용량 문제)
        'data_shapes': {
            'speed_matrix': speed_matrix.shape,
            'adj_matrix': adj_matrix.shape,
            'poi_matrix': poi_matrix.shape,
            'weather_matrix': weather_matrix.shape
        },
        'feature_info': {
            'poi_features': ['평균속도', '속도변동', '최대속도', '최소속도', '평균경도', '평균위도', 
                           '경도변동', '위도변동', '회전선호', '차선선호', '속도위반', '위험경험'],
            'weather_features': ['Period_F', 'Period_A', 'Period_N', 'Period_D', '평일', '주말',
                               '출근시간', '퇴근시간', '점심시간', '시간정규화', '요일정규화', 
                               '날짜정규화', '교통밀도', '네트워크품질']
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {metadata_path}: 메타데이터")

def convert_v2x_to_astgcn_format(data_dir='data/daily_merged/8월', 
                                 output_dir='v2x_astgcn_data',
                                 time_interval='5min',
                                 distance_threshold=500):
    """
    V2X 데이터를 AST-GCN 형식으로 전체 변환
    
    Args:
        data_dir (str): V2X 데이터 디렉토리
        output_dir (str): 출력 디렉토리
        time_interval (str): 시간 간격
        distance_threshold (float): 인접 거리 임계값
    
    Returns:
        tuple: (speed_matrix, poi_matrix, weather_matrix, adj_matrix, vehicle_ids)
    """
    print("=" * 60)
    print("🚗 V2X → AST-GCN 데이터 변환 시작")
    print("=" * 60)
    
    # 1. 데이터 로딩
    raw_df, label_data = load_v2x_data(data_dir)
    
    # 2. 시간×차량 행렬 생성
    speed_matrix, vehicle_ids, time_index = create_time_vehicle_matrix(raw_df, time_interval)
    
    # 3. 인접행렬 생성
    adj_matrix = create_adjacency_matrix(raw_df, vehicle_ids, distance_threshold)
    
    # 4. 정적 속성 생성
    poi_matrix = create_poi_features(raw_df, label_data, vehicle_ids)
    
    # 5. 동적 속성 생성
    weather_matrix = create_weather_features(time_index)
    
    # 6. AST-GCN 형식으로 저장
    save_astgcn_format(speed_matrix, adj_matrix, poi_matrix, weather_matrix,
                       vehicle_ids, time_index, output_dir)
    
    print("=" * 60)
    print("✅ V2X → AST-GCN 변환 완료!")
    print(f"📂 출력 위치: {output_dir}")
    print(f"📊 데이터 요약:")
    print(f"   🚗 차량 수: {len(vehicle_ids)}")
    print(f"   ⏰ 시간 스텝: {len(time_index)}")
    print(f"   🏢 정적 특성: {poi_matrix.shape[1]}개")
    print(f"   🌤️ 동적 특성: {weather_matrix.shape[1]}개")
    print("=" * 60)
    
    return speed_matrix, poi_matrix, weather_matrix, adj_matrix, vehicle_ids

# 메인 실행 부분
if __name__ == "__main__":
    # 설정 가능한 파라미터
    DATA_DIR = "data/daily_merged/08월"  # V2X 데이터 경로
    OUTPUT_DIR = "v2x_astgcn_data"     # 출력 폴더
    TIME_INTERVAL = "15min"             # 시간 간격 (5분)
    DISTANCE_THRESHOLD = 500           # 인접 거리 (500m)
    
    # V2X 데이터가 있는지 확인
    if not os.path.exists(DATA_DIR):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {DATA_DIR}")
        print("💡 경로를 확인하거나 다음과 같이 수정하세요:")
        print("   DATA_DIR = '여러분의/V2X/데이터/경로'")
        exit(1)
    
    try:
        # 변환 실행
        speed_matrix, poi_matrix, weather_matrix, adj_matrix, vehicle_ids = convert_v2x_to_astgcn_format(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            time_interval=TIME_INTERVAL,
            distance_threshold=DISTANCE_THRESHOLD
        )
        
        print("🎉 변환 성공! 이제 다음 단계를 진행하세요:")
        print("   1. acell.py에서 load_v2x_data 함수 추가")
        print("   2. main.py에서 데이터 로딩 부분 수정")
        print("   3. python main.py 실행")
        
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        print("💡 오류 해결 방법:")
        print("   1. 데이터 파일 경로 확인")
        print("   2. CSV 파일 형식 확인 (TRIP_ID, SPEED, TIMESTAMP 컬럼)")
        print("   3. 권한 문제 확인")
        
        import traceback
        traceback.print_exc()