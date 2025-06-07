#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2X 지역 그리드 기반 이상탐지를 위한 acell.py - 클래스 균형 조정 개선버전

핵심 변경점:
- 노드: 차량 → 지역 격자
- 데이터: 개별 차량 이상점수 → 격자별 평균 이상점수
- 안정성: 고정된 그래프 구조
- 클래스 균형: 가중치 계산 및 동적 임계값 적용

Author: V2X Grid-based Anomaly Detection Team  
Date: 2025-06-07
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf  # TF 2.x 호환성
import json
import os
from sklearn.utils.class_weight import compute_class_weight

dim = 20

def load_assist_data(dataset):
    """기존 Shenzhen 데이터 로딩 (호환성을 위해 유지)"""
    sz_adj = pd.read_csv('%s_adj.csv'%dataset, header=None)
    adj = np.array(sz_adj)  # ← np.mat을 np.array로 변경
    data = pd.read_csv('sz_speed.csv')
    return data, adj

def load_v2x_data(dataset='v2x'):
    """
    V2X 지역 그리드 기반 이상탐지 데이터 로딩
    
    Args:
        dataset (str): 데이터셋 이름 (기본값: 'v2x')
    
    Returns:
        data (DataFrame): 격자별 이상점수 데이터 (시간 × 격자)
        adj (matrix): 격자 간 인접행렬 (격자 × 격자)
        poi_data (ndarray): 격자별 정적 속성 (격자 × 특성)
        weather_data (ndarray): 시간별 동적 속성 (시간 × 특성)
    """
    print(f"🗺️ V2X 지역 그리드 이상탐지 데이터 로딩: {dataset}")
    
    # 데이터 폴더 경로
    data_dir = f'v2x_astgcn_data'
    
    # 메타데이터 확인
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            task_type = metadata.get('task_type', 'anomaly_detection')
            node_type = metadata.get('node_type', 'regional_grid')
            print(f"   📋 Task: {task_type}")
            print(f"   🗺️ Node Type: {node_type}")
            
            # 격자 정보 출력
            if 'grid_info' in metadata:
                grid_info = metadata['grid_info']
                print(f"   📊 격자 크기: {grid_info.get('grid_size_meters', 500)}m")
                print(f"   🔢 활성 격자: {grid_info.get('active_grids_used', 'N/A')}개")
    else:
        print("   ⚠️ 메타데이터가 없습니다.")
        task_type = 'anomaly_detection'
        node_type = 'regional_grid'
    
    # 1. 격자 간 인접행렬 로딩
    adj_path = os.path.join(data_dir, f'{dataset}_adj.csv')
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"❌ 격자 인접행렬 파일을 찾을 수 없습니다: {adj_path}")
    
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df) 
    print(f"   ✅ 격자 인접행렬: {adj.shape}")
    
    # 인접행렬 통계
    connection_ratio = np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1])
    avg_connections = np.count_nonzero(adj, axis=1).mean()
    print(f"     📊 연결 비율: {connection_ratio:.3f}")
    print(f"     📊 격자당 평균 연결: {avg_connections:.1f}개")
    
    # 2. 격자별 이상점수 데이터 로딩
    speed_path = os.path.join(data_dir, f'{dataset}_speed.csv')
    if not os.path.exists(speed_path):
        raise FileNotFoundError(f"❌ 이상점수 데이터 파일을 찾을 수 없습니다: {speed_path}")
    
    data = pd.read_csv(speed_path, header=None)
    print(f"   ✅ 격자별 이상점수: {data.shape} (시간 × 격자)")
    
    # 이상점수 데이터 검증
    data_values = data.values
    print(f"     📊 이상점수 범위: {data_values.min():.3f} ~ {data_values.max():.3f}")
    print(f"     📊 평균 이상점수: {data_values.mean():.3f}")
    
    # 이상 비율 계산 (임계값 0.3 기준)
    anomaly_ratio = (data_values > 0.3).mean()
    print(f"     🔥 전체 이상 비율: {anomaly_ratio*100:.2f}%")
    
    # 격자별 이상점수 분포
    grid_anomaly_means = data.mean(axis=0)
    print(f"     📈 격자별 이상점수:")
    print(f"       최소: {grid_anomaly_means.min():.3f}")
    print(f"       최대: {grid_anomaly_means.max():.3f}")
    print(f"       표준편차: {grid_anomaly_means.std():.3f}")
    
    # 3. 격자별 정적 속성 로딩 (POI)
    poi_path = os.path.join(data_dir, f'{dataset}_poi.csv')
    if not os.path.exists(poi_path):
        raise FileNotFoundError(f"❌ 격자 POI 파일을 찾을 수 없습니다: {poi_path}")
    
    poi_df = pd.read_csv(poi_path, header=None)
    poi_data = poi_df.values
    print(f"   ✅ 격자별 정적 속성: {poi_data.shape} (격자 × 특성)")
    
    if task_type == 'anomaly_detection':
        print(f"     📊 격자 특성: [경도, 위도, 중심거리, 도심여부, 교통밀도, 네트워크중요도, ...]")
        # 도심 격자 비율
        if poi_data.shape[1] >= 4:
            downtown_grids = poi_data[:, 3]  # 도심여부
            downtown_ratio = downtown_grids.mean()
            print(f"     🏢 도심 격자 비율: {downtown_ratio*100:.1f}%")
    
    # 4. 시간별 동적 속성 로딩 (Weather)
    weather_path = os.path.join(data_dir, f'{dataset}_weather.csv')
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"❌ Weather 파일을 찾을 수 없습니다: {weather_path}")
    
    weather_df = pd.read_csv(weather_path, header=None)
    weather_data = weather_df.values
    print(f"   ✅ 시간별 동적 속성: {weather_data.shape} (시간 × 특성)")
    
    if task_type == 'anomaly_detection':
        print(f"     📊 시간 특성: [Period_F, Period_A, Period_N, Period_D, ..., 이상위험도, 네트워크품질]")
        # 이상탐지 관련 특성 확인
        if weather_data.shape[1] >= 16:
            anomaly_risks = weather_data[:, -3]  # 이상위험도 (뒤에서 3번째)
            network_qualities = weather_data[:, -2]  # 네트워크품질 (뒤에서 2번째)
            print(f"     🚨 시간별 평균 이상위험도: {anomaly_risks.mean():.3f}")
            print(f"     📡 시간별 평균 네트워크품질: {network_qualities.mean():.3f}")
    
    # 데이터 검증 및 NaN 처리
    if np.isnan(data_values).any():
        print("   ⚠️ 이상점수 데이터에 NaN 발견, 0으로 대체")
        data = data.fillna(0)
    
    if np.isnan(poi_data).any():
        print("   ⚠️ POI 데이터에 NaN 발견, 평균값으로 대체")
        poi_data = pd.DataFrame(poi_data).fillna(pd.DataFrame(poi_data).mean()).values
    
    if np.isnan(weather_data).any():
        print("   ⚠️ Weather 데이터에 NaN 발견, 평균값으로 대체")
        weather_data = pd.DataFrame(weather_data).fillna(pd.DataFrame(weather_data).mean()).values
    
    return data, adj, poi_data, weather_data

def calculate_class_weights_v2x(trainY, method='balanced', verbose=True):
    """
    V2X 이상탐지를 위한 클래스 가중치 계산 - 임계값 수정 버전
    """
    # 🔧 핵심 수정: 안전한 타입 변환
    if isinstance(trainY, np.matrix):
        y_data = np.asarray(trainY)
    else:
        y_data = np.array(trainY)
    
    # 데이터 평평화
    y_flat = y_data.flatten()
    
    # 🔧 핵심 수정: 더 현실적인 동적 임계값 계산
    non_zero_values = y_flat[y_flat > 0]
    if len(non_zero_values) > 0:
        try:
            # 여러 임계값 후보 계산
            p90 = np.percentile(non_zero_values, 90)
            p80 = np.percentile(non_zero_values, 80)
            p75 = np.percentile(non_zero_values, 75)
            mean_std = non_zero_values.mean() + non_zero_values.std()
            
            # 가장 현실적인 임계값 선택 (더 낮은 범위)
            candidates = [p90, p80, p75, mean_std, 0.3, 0.25]
            dynamic_threshold = min([c for c in candidates if c >= 0.1 and c <= 0.6])
            
        except:
            dynamic_threshold = 0.25
        dynamic_threshold = float(dynamic_threshold)
    else:
        dynamic_threshold = 0.25
    
    # 이진 분류를 위한 임계값 적용
    y_binary = (y_flat > dynamic_threshold).astype(int)
    
    # 클래스 개수 계산
    pos_count = np.sum(y_binary == 1)  # 이상 데이터
    neg_count = np.sum(y_binary == 0)  # 정상 데이터
    total_count = len(y_binary)
    
    # 클래스 비율
    pos_ratio = pos_count / total_count
    neg_ratio = neg_count / total_count
    
    if verbose:
        print(f"📊 V2X 클래스 분포 분석 (수정된 임계값: {dynamic_threshold:.3f}):")
        print(f"   전체 샘플: {total_count:,}")
        print(f"   정상 데이터: {neg_count:,} ({neg_ratio:.1%})")
        print(f"   이상 데이터: {pos_count:,} ({pos_ratio:.1%})")
        if pos_count > 0:
            print(f"   불균형 비율: {neg_count/pos_count:.1f}:1")
    
    # 가중치 계산
    if pos_count == 0:
        print("⚠️ 여전히 이상 데이터가 없습니다!")
        print(f"💡 임계값을 더 낮춰보세요: {dynamic_threshold * 0.6:.3f}")
        pos_weight = 1.0
    else:
        if method == 'balanced':
            pos_weight = neg_count / pos_count
        elif method == 'moderate':
            pos_weight = np.sqrt(neg_count / pos_count)
        elif method == 'conservative':
            pos_weight = min(3.0, neg_count / pos_count)
        else:
            pos_weight = 1.0
        
        # 극단적 가중치 방지
        pos_weight = np.clip(pos_weight, 1.5, 20.0)
    
    class_distribution = {
        'total_samples': total_count,
        'positive_samples': pos_count,
        'negative_samples': neg_count,
        'positive_ratio': pos_ratio,
        'negative_ratio': neg_ratio,
        'imbalance_ratio': neg_count / pos_count if pos_count > 0 else float('inf'),
        'dynamic_threshold': dynamic_threshold
    }
    
    if verbose:
        print(f"🎯 계산된 클래스 가중치 ({method}): {pos_weight:.2f}")
    
    return pos_weight, class_distribution

def balance_anomaly_data(trainX, trainY, method='oversample', target_ratio=0.15, verbose=True):
    """
    이상 데이터 균형 맞추기
    
    Args:
        trainX, trainY: 훈련 데이터
        method: 'oversample', 'threshold_adjust' 중 선택
        target_ratio: 목표 이상 데이터 비율
        verbose: 상세 출력 여부
    
    Returns:
        balanced_trainX, balanced_trainY: 균형 맞춘 데이터
    """
    original_shape = trainY.shape
    if verbose:
        print(f"🔄 데이터 균형 조정 시작 - 원본 형태: {original_shape}")
    
    # 현재 이상 비율 계산
    y_flat = trainY.flatten()
    current_anomaly_ratio = (y_flat > 0.3).mean()
    
    if verbose:
        print(f"   현재 이상 비율: {current_anomaly_ratio:.3%}")
        print(f"   목표 이상 비율: {target_ratio:.3%}")
    
    if method == 'threshold_adjust':
        # 임계값 조정으로 이상 비율 맞추기
        if current_anomaly_ratio < target_ratio:
            # 임계값을 낮춰서 이상 데이터 증가
            sorted_values = np.sort(y_flat[y_flat > 0])
            if len(sorted_values) > 0:
                target_idx = max(0, int(len(sorted_values) * (1 - target_ratio / current_anomaly_ratio)))
                new_threshold = sorted_values[target_idx] if target_idx < len(sorted_values) else 0.2
                
                # 새로운 임계값 적용
                adjusted_trainY = trainY.copy()
                adjusted_trainY = np.where(adjusted_trainY > new_threshold, 
                                         adjusted_trainY, 
                                         adjusted_trainY * 0.5)  # 경계값 부드럽게 조정
                
                if verbose:
                    new_ratio = (adjusted_trainY.flatten() > 0.3).mean()
                    print(f"   임계값 조정: {0.3:.3f} → {new_threshold:.3f}")
                    print(f"   조정 후 이상 비율: {new_ratio:.3%}")
                
                return trainX, adjusted_trainY
        
        if verbose:
            print("   임계값 조정 불필요 - 원본 데이터 사용")
        return trainX, trainY
    
    elif method == 'oversample':
        # 시간 단위 오버샘플링 (더 현실적)
        time_steps, num_grids = trainY.shape
        
        # 이상이 많은 시간 스텝 찾기
        time_anomaly_scores = (trainY > 0.3).mean(axis=1)  # 각 시간별 이상 격자 비율
        high_anomaly_times = np.where(time_anomaly_scores > time_anomaly_scores.mean())[0]
        
        if len(high_anomaly_times) > 0:
            # 이상이 많은 시간 스텝 복제
            target_additional = int(time_steps * target_ratio / current_anomaly_ratio) - time_steps
            target_additional = min(target_additional, time_steps // 2)  # 최대 50% 추가
            
            if target_additional > 0:
                # 높은 이상 시간들 중에서 랜덤 선택하여 복제
                selected_times = np.random.choice(high_anomaly_times, 
                                                size=min(target_additional, len(high_anomaly_times)), 
                                                replace=True)
                
                # 약간의 노이즈 추가하여 복제
                noise_scale = 0.01
                additional_X = []
                additional_Y = []
                
                for time_idx in selected_times:
                    if time_idx < len(trainX):
                        # X 데이터에 노이즈 추가
                        noise_X = np.random.normal(0, noise_scale, trainX[time_idx].shape)
                        new_X = trainX[time_idx] + noise_X
                        additional_X.append(new_X)
                        
                        # Y 데이터는 원본 유지 (약간의 스케일링만)
                        scale_factor = np.random.uniform(0.95, 1.05)
                        new_Y = trainY[time_idx] * scale_factor
                        additional_Y.append(new_Y)
                
                if additional_X:
                    balanced_trainX = np.vstack([trainX] + additional_X)
                    balanced_trainY = np.vstack([trainY] + additional_Y)
                    
                    if verbose:
                        new_ratio = (balanced_trainY.flatten() > 0.3).mean()
                        print(f"   오버샘플링 완료:")
                        print(f"     추가된 시간 스텝: {len(additional_X):,}")
                        print(f"     조정 후 형태: {balanced_trainY.shape}")
                        print(f"     조정 후 이상 비율: {new_ratio:.3%}")
                    
                    return balanced_trainX, balanced_trainY
        
        if verbose:
            print("   오버샘플링 조건 미충족 - 원본 데이터 사용")
        return trainX, trainY
    
    # 기본적으로 원본 데이터 반환
    return trainX, trainY

def create_anomaly_threshold_labels(data, method='dynamic', smooth=True, verbose=True):
    """
    이상탐지를 위한 개선된 이진 라벨 생성 - 임계값 수정 버전
    """
    # 🔧 핵심 수정: np.mat를 np.array로 강제 변환
    if hasattr(data, 'values'):
        data_values = data.values
    else:
        data_values = data
    
    # matrix 타입을 array로 변환
    if isinstance(data_values, np.matrix):
        data_values = np.asarray(data_values)
    
    # 안전한 타입 변환
    data_values = np.array(data_values, dtype=np.float32)
    
    if verbose:
        print(f"   📊 데이터 타입 변환: {type(data)} → {type(data_values)}")
        print(f"   📊 데이터 형태: {data_values.shape}")
        print(f"   📊 데이터 범위: {data_values.min():.3f} ~ {data_values.max():.3f}")
        print(f"   📊 데이터 평균: {data_values.mean():.3f}")
    
    if method == 'dynamic':
        # 🔧 핵심 수정: 더 현실적인 임계값 계산
        non_zero_mask = data_values > 0
        non_zero_values = data_values[non_zero_mask]
        
        if len(non_zero_values) > 0:
            try:
                non_zero_flat = non_zero_values.flatten()
                
                # 🎯 새로운 임계값 전략
                # 1. 상위 10% 기준 (더 많은 이상 데이터 포함)
                threshold_90 = np.percentile(non_zero_flat, 90)
                
                # 2. 상위 20% 기준 (균형잡힌 접근)
                threshold_80 = np.percentile(non_zero_flat, 80)
                
                # 3. 평균 + 표준편차 기준
                mean_val = non_zero_flat.mean()
                std_val = non_zero_flat.std()
                threshold_stat = mean_val + std_val
                
                # 🔧 가장 적절한 임계값 선택 (더 낮은 값)
                threshold_candidates = [threshold_90, threshold_80, threshold_stat, 0.3]
                threshold = min([t for t in threshold_candidates if t > 0.1])
                
                # 안전 범위 확보
                threshold = max(0.15, min(0.6, threshold))
                
                if verbose:
                    print(f"   🎯 임계값 후보들:")
                    print(f"     90th percentile: {threshold_90:.3f}")
                    print(f"     80th percentile: {threshold_80:.3f}")
                    print(f"     mean + std: {threshold_stat:.3f}")
                    print(f"     선택된 임계값: {threshold:.3f}")
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Percentile 계산 실패: {e}")
                # 폴백: 고정 임계값
                threshold = 0.25
        else:
            threshold = 0.25
            
    elif method == 'percentile':
        # 전체 데이터 기준 상위 15% (더 공격적)
        try:
            data_flat = data_values.flatten()
            threshold = np.percentile(non_zero_flat, 80)  # 더 많은 이상 데이터 포함
            threshold = max(0.2, min(0.4, threshold))     # 안전 범위 보장
        except:
            threshold = 0.25
    else:  # fixed
        threshold = 0.25  # 기본값을 0.3에서 0.25로 낮춤
    
    # 임계값을 scalar로 확실히 변환
    threshold = float(threshold)
    
    binary_labels = (data_values > threshold).astype(float)
    
    if smooth:
        # 시간적 스무딩 (연속된 이상 패턴 강화)
        try:
            from scipy import ndimage
            
            # 각 격자별로 시간 축 스무딩
            for grid_idx in range(binary_labels.shape[1]):
                grid_series = binary_labels[:, grid_idx]
                
                # 3점 이동평균으로 스무딩
                smoothed = ndimage.uniform_filter1d(grid_series.astype(float), size=3)
                binary_labels[:, grid_idx] = (smoothed > 0.3).astype(float)
        except ImportError:
            if verbose:
                print("   ⚠️ scipy 없음 - 스무딩 건너뛰기")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 스무딩 실패: {e}")
    
    if verbose:
        anomaly_ratio = binary_labels.mean()
        print(f"   📊 임계값 방법: {method}")
        print(f"   📊 최종 사용된 임계값: {threshold:.3f}")
        print(f"   📊 생성된 이상 비율: {anomaly_ratio:.3%}")
        
        # 추가 검증 정보
        if anomaly_ratio == 0:
            print(f"   ⚠️ 여전히 이상 데이터가 없습니다!")
            print(f"   💡 더 낮은 임계값 시도: {threshold * 0.7:.3f}")
        elif anomaly_ratio > 0.5:
            print(f"   ⚠️ 이상 데이터가 너무 많습니다 ({anomaly_ratio:.1%})")
            print(f"   💡 더 높은 임계값 시도: {threshold * 1.3:.3f}")
        else:
            print(f"   ✅ 적절한 이상 비율 달성!")
    
    return binary_labels, threshold

def create_anomaly_threshold_labels_stable(data, threshold=0.3, verbose=True):
    """
    안정화된 이상탐지 라벨 생성 - 고정 임계값 사용
    """
    # 안전한 데이터 변환
    if hasattr(data, 'values'):
        data_values = data.values
    else:
        data_values = data
    
    if isinstance(data_values, np.matrix):
        data_values = np.asarray(data_values)
    
    data_values = np.array(data_values, dtype=np.float32)
    
    if verbose:
        print(f"   📊 안정화된 라벨 생성:")
        print(f"     데이터 형태: {data_values.shape}")
        print(f"     데이터 범위: {data_values.min():.3f} ~ {data_values.max():.3f}")
        print(f"     고정 임계값: {threshold:.3f}")
    
    # NaN/Inf 처리
    data_values = np.nan_to_num(data_values, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 고정 임계값으로 이진화
    binary_labels = (data_values > threshold).astype(float)
    
    if verbose:
        anomaly_ratio = binary_labels.mean()
        print(f"     생성된 이상 비율: {anomaly_ratio:.3%}")
        
        if anomaly_ratio == 0:
            print(f"     ⚠️ 이상 데이터가 없습니다!")
            print(f"     💡 임계값을 낮춰보세요: {threshold * 0.7:.3f}")
        elif anomaly_ratio > 0.5:
            print(f"     ⚠️ 이상 데이터가 너무 많습니다")
            print(f"     💡 임계값을 높여보세요: {threshold * 1.3:.3f}")
        else:
            print(f"     ✅ 적절한 이상 비율!")
    
    return binary_labels, threshold

def calculate_class_weights_stable(trainY, threshold=0.3, max_weight=3.0, verbose=True):
    """
    안정화된 클래스 가중치 계산 - 가중치 제한
    """
    # 안전한 타입 변환
    if isinstance(trainY, np.matrix):
        y_data = np.asarray(trainY)
    else:
        y_data = np.array(trainY)
    
    y_flat = y_data.flatten()
    
    # 고정 임계값으로 이진화
    y_binary = (y_flat > threshold).astype(int)
    
    # 클래스 개수 계산
    pos_count = np.sum(y_binary == 1)
    neg_count = np.sum(y_binary == 0)
    total_count = len(y_binary)
    
    if verbose:
        print(f"📊 안정화된 클래스 분포 (고정 임계값: {threshold:.3f}):")
        print(f"   전체 샘플: {total_count:,}")
        print(f"   정상 데이터: {neg_count:,} ({neg_count/total_count:.1%})")
        print(f"   이상 데이터: {pos_count:,} ({pos_count/total_count:.1%})")
    
    # 가중치 계산 (제한 적용)
    if pos_count == 0:
        print("⚠️ 이상 데이터가 없습니다!")
        pos_weight = 1.0
    else:
        raw_weight = neg_count / pos_count
        pos_weight = min(max_weight, max(1.5, raw_weight))  # 1.5~3.0 범위
        
        if verbose:
            print(f"   원시 가중치: {raw_weight:.2f}")
            print(f"   제한된 가중치: {pos_weight:.2f} (최대 {max_weight})")
    
    return pos_weight

def preprocess_data_stable(data1, time_len, train_rate, seq_len, pre_len, 
                          model_name, scheme, poi_data=None, weather_data=None,
                          threshold=0.3):
    """
    안정화된 V2X 이상탐지 전처리 - 일관된 데이터 처리
    """
    print(f"🛠️ 안정화된 V2X 이상탐지 전처리:")
    print(f"   📊 데이터 형태: {data1.shape}")
    print(f"   🎯 고정 임계값: {threshold}")
    print(f"   🔧 시퀀스 길이: {seq_len}, 예측 길이: {pre_len}")
    
    # 안전한 데이터 변환
    if isinstance(data1, np.matrix):
        data1 = np.asarray(data1)
    
    data_values = np.array(data1, dtype=np.float32)
    
    # NaN/Inf 처리
    data_values = np.nan_to_num(data_values, nan=0.0, posinf=1.0, neginf=0.0)
    
    print(f"   📊 데이터 통계:")
    print(f"     범위: {data_values.min():.3f} ~ {data_values.max():.3f}")
    print(f"     평균: {data_values.mean():.3f}")
    
    # 🎯 핵심: 연속값과 이진값 동일한 임계값 사용
    binary_labels, _ = create_anomaly_threshold_labels_stable(
        data_values, threshold=threshold, verbose=True
    )
    
    # 훈련/테스트 분할
    train_size = int(time_len * train_rate)
    
    # 🎯 중요: 라벨도 연속값 사용 (일관성 확보)
    train_data = data_values[:train_size]
    test_data = data_values[train_size:]
    
    train_labels = binary_labels[:train_size]
    test_labels = binary_labels[train_size:]
    
    print(f"   ✂️ 분할 완료:")
    print(f"     훈련: {train_data.shape}")
    print(f"     테스트: {test_data.shape}")
    
    # 시퀀스 생성 (단순화)
    trainX, trainY, testX, testY = [], [], [], []
    
    # 훈련 시퀀스
    for i in range(seq_len, len(train_data) - pre_len + 1):
        # 입력: 연속값
        seq_x = train_data[i-seq_len:i].T  # (nodes, seq_len)
        # 라벨: 이진값
        seq_y = train_labels[i:i+pre_len].T  # (nodes, pre_len)
        
        trainX.append(seq_x)
        trainY.append(seq_y)
    
    # 테스트 시퀀스
    for i in range(seq_len, len(test_data) - pre_len + 1):
        seq_x = test_data[i-seq_len:i].T
        seq_y = test_labels[i:i+pre_len].T
        
        testX.append(seq_x)
        testY.append(seq_y)
    
    # 배열 변환
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    
    # 차원 조정: (samples, seq_len, nodes)
    trainX = np.transpose(trainX, (0, 2, 1))
    trainY = np.transpose(trainY, (0, 2, 1))
    testX = np.transpose(testX, (0, 2, 1))
    testY = np.transpose(testY, (0, 2, 1))
    
    # 안정화된 클래스 가중치 계산
    pos_weight = calculate_class_weights_stable(
        trainY, threshold=threshold, max_weight=3.0, verbose=True
    )
    
    # 최종 검증
    train_anomaly_ratio = (trainY > threshold).mean()
    test_anomaly_ratio = (testY > threshold).mean()
    
    print(f"   ✅ 안정화 완료:")
    print(f"     trainX: {trainX.shape}")
    print(f"     trainY: {trainY.shape}")
    print(f"     testX: {testX.shape}")
    print(f"     testY: {testY.shape}")
    print(f"     클래스 가중치: {pos_weight:.2f}")
    print(f"     훈련 이상 비율: {train_anomaly_ratio:.2%}")
    print(f"     테스트 이상 비율: {test_anomaly_ratio:.2%}")
    
    # 균형 정보 반환
    balance_info = {
        'pos_weight': pos_weight,
        'threshold_used': threshold,
        'train_anomaly_ratio': train_anomaly_ratio,
        'test_anomaly_ratio': test_anomaly_ratio
    }
    
    return trainX, trainY, testX, testY, balance_info

# 기존 preprocess_data 함수를 안정화된 버전으로 대체
def preprocess_data(data1, time_len, train_rate, seq_len, pre_len, model_name, scheme, poi_data=None, weather_data=None):
    """
    V2X 데이터 전처리 (안정화된 버전으로 리다이렉트)
    """
    print("🔄 안정화된 전처리 함수로 리다이렉트...")
    return preprocess_data_stable(
        data1, time_len, train_rate, seq_len, pre_len,
        model_name, scheme, poi_data, weather_data,
        threshold=0.3  # 고정 임계값 사용
    )

# 기존 preprocess_data_grid_anomaly 함수도 안정화된 버전으로 대체
def preprocess_data_grid_anomaly(data1, time_len, train_rate, seq_len, pre_len, model_name, scheme, poi_data=None, weather_data=None):
    """
    지역 그리드 기반 이상탐지 데이터 전처리 (안정화된 버전으로 리다이렉트)
    """
    print("🔄 안정화된 전처리 함수로 리다이렉트...")
    return preprocess_data_stable(
        data1, time_len, train_rate, seq_len, pre_len,
        model_name, scheme, poi_data, weather_data,
        threshold=0.3  # 고정 임계값 사용
    )

# 기존 Unit 클래스들은 그대로 유지 (변경 없음)
class Unit():
    def __init__(self, dim, num_nodes, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        x_output = tf.add(x1, self.bias_unit)
        return x_output
    
    def _emb(self, dim, time_len):
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            bias_unit = tf.get_variable(name = 'bias_unit', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w = weight_unit
        b = bias_unit
        return w, b

class Unit1():
    def __init__(self, dim, num_nodes, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix=tf.convert_to_tensor(unit_matrix1)
        self.weight_unit1 ,self.bias_unit1 = self._emb(dim, time_len)
        
        x1=tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit1)
        x_output= tf.add(x1, self.bias_unit1)
        return x_output
    
    def _emb(self, dim, time_len):
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit1 = tf.get_variable(name = 'weight_unit1', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            bias_unit1 = tf.get_variable(name = 'bias_unit1', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w1 = weight_unit1
        bb = bias_unit1
        return w1, bb

class Unit2():
    def __init__(self, dim, num_nodes, time_len, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes
        self.time_len = time_len
    def call(self, inputs, time_len):
        x, e = inputs
        x = np.transpose(x)
        x = x.astype(np.float64)
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        self.x_output = tf.add(x1, self.bias_unit)
        self.x_output = tf.transpose(self.x_output)
        return self.x_output
    
    def _emb(self, dim, time_len):
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            self.weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.time_len), dtype = tf.float32)
            self.bias_unit = tf.get_variable(name = 'bias_unit', shape =(self.num_nodes,1), initializer = tf.constant_initializer(dtype=tf.float32))
        self.w = self.weight_unit
        self.b = self.bias_unit
        return self.w, self.b

class Unit3():
    def __init__(self, dim, num_nodes, time_len, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes
        self.time_len = time_len
    def call(self, inputs, time_len):
        x, e = inputs
        x = np.transpose(x)
        x = x.astype(np.float64)
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        x_output = tf.add(x1, self.bias_unit)
        x_output = tf.transpose(x_output)
        return x_output
    
    def _emb(self, dim, time_len):
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.time_len), dtype = tf.float32)
            bias_unit = tf.get_variable(name = 'bias_unit', shape =(self.num_nodes,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w = weight_unit
        b = bias_unit
        return w, b

class Unit4():
    def __init__(self, dim, num_nodes, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        x_output = tf.add(x1, self.bias_unit)
        return x_output
    
    def _emb(self, dim, time_len):
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            bias_unit = tf.get_variable(name = 'bias_unit', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w = weight_unit
        b = bias_unit
        return w, b

class Unit5():
    def __init__(self, dim, num_nodes, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix=tf.convert_to_tensor(unit_matrix1)
        self.weight_unit1 ,self.bias_unit1 = self._emb(dim, time_len)
        
        x1=tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit1)
        self.x_output= tf.add(x1, self.bias_unit1)
        return self.x_output
    
    def _emb(self, dim, time_len):
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            self.weight_unit1 = tf.get_variable(name = 'weight_unit1', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            self.bias_unit1 = tf.get_variable(name = 'bias_unit1', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        self.w1 = self.weight_unit1
        self.bb = self.bias_unit1
        return self.w1, self.bb

print("✅ 개선된 acell.py 로드 완료!")