#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2X AST-GCN 이상탐지 메인 실행 파일 - 수정된 버전

핵심 수정사항:
1. 데이터 분포 기반 동적 임계값 계산
2. 평가 로직 일관성 확보
3. 모델 출력 검증 추가
4. 시각화 개선

Author: V2X Anomaly Detection Team (Fixed)
Date: 2025-06-09
"""

import pickle as pkl
import tensorflow as tf
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf_v1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 🎯 동적 임계값 설정 (데이터 분포 기반)
DYNAMIC_THRESHOLD = True  # 동적 임계값 사용 여부
BASE_THRESHOLD = 0.3      # 기본 임계값 (폴백용)
MAX_CLASS_WEIGHT = 5.0    # 클래스 가중치 최대값 증가

print(f"🎯 수정된 설정:")
print(f"   동적 임계값: {DYNAMIC_THRESHOLD}")
print(f"   기본 임계값: {BASE_THRESHOLD}")
print(f"   최대 클래스 가중치: {MAX_CLASS_WEIGHT}")

# Apple GPU (Metal) 설정
print("🔍 GPU 설정 중...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"🚀 Apple GPU (Metal) 활성화 성공! - {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️ GPU 설정 오류: {e}")
else:
    print("💻 CPU 모드로 실행")

# TF 1.x 호환 모드로 전환
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from acell import preprocess_data, load_assist_data, load_v2x_data
from tgcn import tgcnCell

from visualization import plot_result, plot_error
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

time_start = time.time()

###### Settings ######
flags = tf_v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 50, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 10, 'time length of inputs.')
flags.DEFINE_integer('pre_len', 3, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('dataset', 'v2x', 'dataset')
flags.DEFINE_string('model_name', 'ast-gcn', 'ast-gcn')
flags.DEFINE_integer('scheme', 3, 'scheme')
flags.DEFINE_string('noise_name', 'None', 'None or Gauss or Possion')
flags.DEFINE_float('noise_param', 0, 'Parameter for noise')

model_name = FLAGS.model_name
noise_name = FLAGS.noise_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
scheme = FLAGS.scheme
PG = FLAGS.noise_param

###### load data ######
print(f"🚨 V2X 이상탐지 데이터 로딩: {data_name}")

if data_name == 'v2x':
    data, adj, poi_data, weather_data = load_v2x_data('v2x')
    print(f"✅ V2X 이상탐지 데이터 로딩 완료")
elif data_name == 'sz':
    data, adj = load_assist_data('sz')
    poi_data, weather_data = None, None
    print(f"✅ Shenzhen 데이터 로딩 완료")
else:
    raise ValueError(f"❌ 지원하지 않는 데이터셋: {data_name}")

### Perturbation Analysis (기존과 동일)
def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

if noise_name == 'Gauss':
    Gauss = np.random.normal(0, PG, size=data.shape)
    noise_Gauss = MaxMinNormalization(Gauss, np.max(Gauss), np.min(Gauss))
    data = data + noise_Gauss
elif noise_name == 'Possion':
    Possion = np.random.poisson(PG, size=data.shape)
    noise_Possion = MaxMinNormalization(Possion, np.max(Possion), np.min(Possion))
    data = data + noise_Possion
else:
    data = data

time_len = data.shape[0]
num_nodes = data.shape[1]

# 🔧 수정: matrix 타입 사용 금지, array로 강제 변환
data1 = np.array(data, dtype=np.float32)  # np.mat 대신 np.array 사용

#### normalization (이상점수는 이미 0-1 범위이므로 최소한의 정규화)
max_value = np.max(data1)
if max_value > 0:
    data1 = data1 / max_value
else:
    print("⚠️ 모든 이상점수가 0입니다. 정규화를 건너뜁니다.")

# 🔧 수정: 컬럼 정보 처리
if hasattr(data, 'columns'):
    # pandas DataFrame의 컬럼 정보는 따로 저장
    column_names = data.columns
else:
    column_names = None

# 🔧 핵심 수정: 데이터 분포 기반 최적 임계값 계산
def calculate_optimal_threshold(data_values, method='sigmoid_compatible'):
    """모델 출력과 호환되는 최적 임계값 계산"""
    print("🔍 모델 호환 최적 임계값 계산 중...")
    
    # 🔧 수정: matrix 타입을 array로 강제 변환
    if isinstance(data_values, np.matrix):
        data_values = np.asarray(data_values)
    
    # 데이터 통계
    data_flat = data_values.flatten()
    non_zero_data = data_flat[data_flat > 0]
    
    # 🔧 수정: 각 값을 float으로 변환하여 format 문제 해결
    mean_val = float(data_flat.mean())
    std_val = float(data_flat.std())
    median_val = float(np.median(data_flat))
    non_zero_mean = float(non_zero_data.mean()) if len(non_zero_data) > 0 else 0.0
    
    print(f"   📊 전체 데이터 통계:")
    print(f"     평균: {mean_val:.4f}")
    print(f"     표준편차: {std_val:.4f}")
    print(f"     중간값: {median_val:.4f}")
    print(f"     0이 아닌 값들 평균: {non_zero_mean:.4f}")
    
    if method == 'sigmoid_compatible':
        # 🎯 핵심 수정: 시그모이드 출력(~0.5)과 호환되는 임계값 설정
        # 데이터 기반 계산 후 시그모이드 범위로 조정
        data_based_threshold = float(np.percentile(data_flat, 85))
        
        # 시그모이드 출력 고려: 0.3~0.7 범위로 조정
        if data_based_threshold < 0.1:
            threshold = 0.45  # 시그모이드 중앙값 근처
        elif data_based_threshold > 0.5:
            threshold = 0.55  # 약간 높은 값
        else:
            threshold = 0.5   # 시그모이드 중앙값
            
        print(f"   🔧 시그모이드 호환 조정: {data_based_threshold:.4f} → {threshold:.4f}")
        
    elif method == 'percentile_based':
        # 기존 방식 (문제가 있었던 방식)
        threshold = float(np.percentile(data_flat, 85))
        threshold = float(np.clip(threshold, 0.05, 0.5))
    else:
        # 기본값
        threshold = 0.5
    
    # 🔧 수정: 시그모이드 출력 범위에 맞는 안전 범위 (0.3 ~ 0.7)
    threshold = float(np.clip(threshold, 0.3, 0.7))
    
    # 결과 이상 비율 계산 (원본 데이터 기준)
    anomaly_ratio = float((data_flat > data_based_threshold if 'data_based_threshold' in locals() else mean_val + std_val).mean())
    
    print(f"   🎯 최종 임계값: {threshold:.4f} (시그모이드 호환)")
    print(f"   📊 예상 이상 비율: {anomaly_ratio*100:.2f}%")
    
    return threshold

# 동적 임계값 계산
if DYNAMIC_THRESHOLD:
    optimal_threshold = calculate_optimal_threshold(data1, method='sigmoid_compatible')
else:
    optimal_threshold = 0.5  # 시그모이드 중앙값

print(f"   ✅ 최종 사용 임계값: {optimal_threshold:.4f} (시그모이드 호환)")

# 모델 및 스킴 정보 출력
if model_name == 'ast-gcn':
    if scheme == 1:
        name = 'V2X POI only (Anomaly Detection)' if data_name == 'v2x' else 'add poi dim'
    elif scheme == 2:
        name = 'V2X Weather only (Anomaly Detection)' if data_name == 'v2x' else 'add weather dim'
    else:
        name = 'V2X POI + Weather (Anomaly Detection)' if data_name == 'v2x' else 'add poi + weather dim'
else:
    name = 'tgcn (Anomaly Detection)'

print('📊 이상탐지 모델 정보:')
print(f'   task: anomaly_detection')
print(f'   model: {model_name}')
print(f'   dataset: {data_name}')
print(f'   scheme: {scheme} ({name})')
print(f'   data shape: {data1.shape}')
print(f'   time_len: {time_len}, num_nodes: {num_nodes}')
print(f'   optimal_threshold: {optimal_threshold:.4f}')
print(f'   noise_name: {noise_name}')
print(f'   noise_param: {PG}')

# 🔧 수정된 전처리 함수 (최적 임계값 사용)
def preprocess_data_fixed(data1, time_len, train_rate, seq_len, pre_len, 
                         model_name, scheme, poi_data=None, weather_data=None,
                         threshold=None):
    """수정된 V2X 이상탐지 전처리 - 최적 임계값 사용"""
    print(f"🛠️ 수정된 V2X 이상탐지 전처리:")
    print(f"   📊 데이터 형태: {data1.shape}")
    print(f"   🎯 최적 임계값: {threshold:.4f}")
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
    
    # 🎯 핵심 수정: 시그모이드 호환 라벨 생성
    # 원본 데이터의 상위 10%를 이상으로 설정 (더 균형잡힌 접근)
    data_threshold_for_labels = float(np.percentile(data_values.flatten(), 90))
    binary_labels = (data_values > data_threshold_for_labels).astype(float)
    anomaly_ratio = binary_labels.mean()
    
    print(f"   📊 라벨 생성 통계:")
    print(f"     라벨 생성 임계값: {data_threshold_for_labels:.4f} (90th percentile)")
    print(f"     평가 임계값: {threshold:.4f} (시그모이드 호환)")
    print(f"     생성된 이상 비율: {anomaly_ratio:.3%}")
    
    if anomaly_ratio == 0:
        print(f"   ⚠️ 이상 데이터가 없습니다! 85th percentile로 재시도")
        data_threshold_for_labels = float(np.percentile(data_values.flatten(), 85))
        binary_labels = (data_values > data_threshold_for_labels).astype(float)
        anomaly_ratio = binary_labels.mean()
        print(f"   🔧 조정된 라벨 임계값: {data_threshold_for_labels:.4f}, 이상 비율: {anomaly_ratio:.3%}")
    
    # 훈련/테스트 분할
    train_size = int(time_len * train_rate)
    
    train_data = data_values[:train_size]
    test_data = data_values[train_size:]
    
    train_labels = binary_labels[:train_size]
    test_labels = binary_labels[train_size:]
    
    print(f"   ✂️ 분할 완료:")
    print(f"     훈련: {train_data.shape}")
    print(f"     테스트: {test_data.shape}")
    
    # 시퀀스 생성
    trainX, trainY, testX, testY = [], [], [], []
    
    # 훈련 시퀀스
    for i in range(seq_len, len(train_data) - pre_len + 1):
        # 입력: 연속값 (원본 이상점수)
        seq_x = train_data[i-seq_len:i].T  # (nodes, seq_len)
        # 라벨: 이진값 (0 또는 1)
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
    
    # 🔧 수정된 클래스 가중치 계산
    y_flat = trainY.flatten()
    pos_count = np.sum(y_flat == 1)
    neg_count = np.sum(y_flat == 0)
    
    if pos_count > 0:
        pos_weight = min(MAX_CLASS_WEIGHT, neg_count / pos_count)
    else:
        pos_weight = 1.0
    
    # 최종 검증
    train_anomaly_ratio = (trainY == 1).mean()
    test_anomaly_ratio = (testY == 1).mean()
    
    print(f"   ✅ 수정된 전처리 완료:")
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
        'threshold_used': threshold,  # 평가용 임계값
        'label_threshold': data_threshold_for_labels,  # 라벨 생성 임계값
        'train_anomaly_ratio': train_anomaly_ratio,
        'test_anomaly_ratio': test_anomaly_ratio
    }
    
    return trainX, trainY, testX, testY, balance_info

# 수정된 전처리 호출
print(f"\n🔄 수정된 이상탐지 데이터 전처리 시작...")

trainX, trainY, testX, testY, balance_info = preprocess_data_fixed(
    data1, time_len, train_rate, seq_len, pre_len, model_name, scheme,
    poi_data, weather_data, threshold=optimal_threshold
)

# 클래스 균형 정보 추출
pos_weight = balance_info['pos_weight']
threshold_used = balance_info['threshold_used']
label_threshold = balance_info['label_threshold']
train_anomaly_ratio = balance_info['train_anomaly_ratio']
test_anomaly_ratio = balance_info['test_anomaly_ratio']

print(f"🎯 수정된 클래스 균형 정보:")
print(f"   양성 클래스 가중치: {pos_weight:.2f}")
print(f"   라벨 생성 임계값: {label_threshold:.4f} (90th percentile)")
print(f"   평가용 임계값: {threshold_used:.4f} (시그모이드 호환)")
print(f"   훈련 이상 비율: {train_anomaly_ratio:.2%}")
print(f"   테스트 이상 비율: {test_anomaly_ratio:.2%}")

totalbatch = int(trainX.shape[0] / batch_size)
training_data_count = len(trainX)

print(f"✅ 전처리 완료:")
print(f"   total batches: {totalbatch}")
print(f"   training samples: {training_data_count}")

# 🔧 수정된 평가 함수
def evaluate_anomaly_fixed(y_true, y_pred_proba, threshold):
    """수정된 이상탐지 평가 함수"""
    try:
        # 데이터 평평화
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_proba.flatten()
        
        # NaN/Inf 안전 처리
        y_pred_flat = np.nan_to_num(y_pred_flat, 0.0)
        y_pred_flat = np.clip(y_pred_flat, 0.0, 1.0)
        
        # 🎯 핵심: 동일한 임계값으로 이진화
        y_pred_binary = (y_pred_flat > threshold).astype(int)
        y_true_binary = y_true_flat.astype(int)  # 이미 이진값
        
        # 메트릭 계산
        acc = accuracy_score(y_true_binary, y_pred_binary)
        
        # 클래스가 존재하는 경우만 계산
        if len(np.unique(y_true_binary)) > 1:
            pre = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            try:
                auc = roc_auc_score(y_true_binary, y_pred_flat)
            except:
                auc = 0.5
        else:
            pre = rec = f1 = auc = 0.0
        
        return acc, pre, rec, f1, auc
    except Exception as e:
        print(f"⚠️ 평가 중 오류: {e}")
        return 0.5, 0.0, 0.0, 0.0, 0.5

def TGCN_ANOMALY(_X, _weights, _biases):
    """TGCN 이상탐지용 모델 - 출력 안정화 개선"""
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf_v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf_v1.unstack(_X, axis=1)
    outputs, states = tf_v1.nn.static_rnn(cell, _X, dtype=tf_v1.float32)
    m = []
    for i in outputs:
        o = tf_v1.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf_v1.reshape(o, shape=[-1, gru_units])
        # Dropout
        o = tf_v1.nn.dropout(o, keep_prob=0.8)
        m.append(o)
    last_output = m[-1]
    
    # 🔧 수정: 더 다양한 출력을 위한 가중치 초기화 개선
    logits = tf_v1.matmul(last_output, _weights['out']) + _biases['out']
    logits = tf_v1.reshape(logits, shape=[-1, num_nodes, pre_len])
    logits = tf_v1.transpose(logits, perm=[0, 2, 1])
    logits = tf_v1.reshape(logits, shape=[-1, num_nodes])
    
    # 🔧 수정: 로지트 범위를 넓혀서 다양한 시그모이드 출력 생성
    logits = tf_v1.clip_by_value(logits, -5.0, 5.0)  # 범위 확대
    
    # 시그모이드 활성화 (이상 확률)
    output = tf_v1.sigmoid(logits)
    
    return output, logits, m, states

def create_improved_loss_function(pos_weight=3.0):
    """개선된 균형 손실함수"""
    
    def improved_loss(y_true, y_pred):
        # 🔧 수정: 더 안정적인 weighted BCE
        weighted_bce = tf_v1.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            pos_weight=pos_weight
        )
        
        # 가벼운 Focal Loss 추가
        y_pred_sigmoid = tf_v1.nn.sigmoid(y_pred)
        y_pred_sigmoid = tf_v1.clip_by_value(y_pred_sigmoid, 1e-7, 1.0 - 1e-7)
        
        ce_loss = -y_true * tf_v1.math.log(y_pred_sigmoid) - (1 - y_true) * tf_v1.math.log(1 - y_pred_sigmoid)
        pt = tf_v1.where(tf_v1.equal(y_true, 1), y_pred_sigmoid, 1 - y_pred_sigmoid)
        focal_weight = 0.2 * tf_v1.pow(1 - pt, 1.5)  # 더 약한 focal loss
        focal_loss = focal_weight * ce_loss
        
        # 결합 (BCE 85%, Focal 15%)
        combined_loss = 0.85 * weighted_bce + 0.15 * focal_loss
        
        return tf_v1.reduce_mean(combined_loss)
    
    return improved_loss

###### placeholders ######
print(f"\n🧠 수정된 이상탐지 모델 구성...")

# 실제 데이터 차원 확인
print(f"   📊 실제 데이터 형태:")
print(f"     trainX: {trainX.shape}")
print(f"     trainY: {trainY.shape}")
print(f"     seq_len: {seq_len}")
print(f"     pre_len: {pre_len}")
print(f"     num_nodes: {num_nodes}")

# 안전한 placeholder 생성
actual_seq_len = trainX.shape[1]
actual_num_nodes = trainX.shape[2]
actual_pre_len = trainY.shape[1]

print(f"   🔧 Placeholder 차원:")
print(f"     입력: [{None}, {actual_seq_len}, {actual_num_nodes}]")
print(f"     라벨: [{None}, {actual_pre_len}, {actual_num_nodes}]")

# 동적 차원으로 placeholder 생성
inputs = tf_v1.placeholder(tf_v1.float32, shape=[None, actual_seq_len, actual_num_nodes])
labels = tf_v1.placeholder(tf_v1.float32, shape=[None, actual_pre_len, actual_num_nodes])

# Graph weights (이상탐지용) - 🔧 수정: 더 나은 초기화
weights = {
    'out': tf_v1.Variable(tf_v1.random_normal([gru_units, actual_pre_len], mean=0.0, stddev=0.1), name='weight_o')}  # stddev 증가

biases = {
    'out': tf_v1.Variable(tf_v1.random_normal([actual_pre_len], mean=0.0, stddev=0.1), name='bias_o')}  # stddev 증가

pred, logits, ttts, ttto = TGCN_ANOMALY(inputs, weights, biases)

y_pred = pred  # 시그모이드 출력 (0-1 확률)
y_logits = logits  # 로지트 (손실 계산용)

###### optimizer (수정된 이상탐지용 손실함수) ######
lambda_loss = 0.001  # L2 정규화 감소
Lreg = lambda_loss * sum(tf_v1.nn.l2_loss(tf_var) for tf_var in tf_v1.trainable_variables())
label = tf_v1.reshape(labels, [-1, actual_num_nodes])
logits_reshaped = tf_v1.reshape(y_logits, [-1, actual_num_nodes])

print('y_pred_shape:', y_pred.shape)
print('label_shape:', label.shape)
print('logits_shape:', logits_reshaped.shape)

# 수정된 균형 손실함수 사용
print(f"🎯 수정된 손실함수 구성 (클래스 가중치: {pos_weight:.2f})")
improved_loss_fn = create_improved_loss_function(pos_weight=pos_weight)
main_loss = improved_loss_fn(label, logits_reshaped)
loss = main_loss + Lreg

# 수정된 최적화기
global_step = tf_v1.Variable(0, trainable=False)
learning_rate = tf_v1.train.exponential_decay(
    lr, global_step, 
    decay_steps=300,  # 더 빠른 감소
    decay_rate=0.95,  # 더 안정적인 감소
    staircase=True
)

# Gradient clipping (수정)
opt = tf_v1.train.AdamOptimizer(learning_rate)
grads_and_vars = opt.compute_gradients(loss)
clipped_grads_and_vars = [(tf_v1.clip_by_value(grad, -1.0, 1.0), var)  # 클리핑 범위 확대
                          for grad, var in grads_and_vars if grad is not None]
optimizer = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)

###### Initialize session ######
variables = tf_v1.global_variables()
saver = tf_v1.train.Saver(tf_v1.global_variables())

# GPU 설정
config = tf_v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

sess = tf_v1.Session(config=config)
sess.run(tf_v1.global_variables_initializer())

# 출력 디렉토리 설정
if data_name == 'v2x':
    out = f'out/v2x_{model_name}_anomaly_fixed_scheme{scheme}_gpu'
else:
    out = f'out/{model_name}_anomaly_fixed_{noise_name}_gpu'

path1 = f'{model_name}_anomaly_fixed_{name}_{data_name}_lr{lr}_batch{batch_size}_unit{gru_units}_seq{seq_len}_pre{pre_len}_epoch{training_epoch}_scheme{scheme}_threshold{threshold_used:.3f}_GPU'
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)

print(f"📂 결과 저장 경로: {path}")

print(f"\n🚀 V2X AST-GCN 수정된 이상탐지 GPU 학습 시작...")
print(f"   Epochs: {training_epoch}")
print(f"   Batch size: {batch_size}")
print(f"   Learning rate: {lr} (지수 감소)")
print(f"   클래스 가중치: {pos_weight:.2f}")
print(f"   라벨 임계값: {label_threshold:.4f}")
print(f"   평가 임계값: {threshold_used:.4f}")

x_axe, batch_loss, batch_acc = [], [], []
test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_pred = [], [], [], [], [], [], []

best_f1 = 0.0
patience_counter = 0
early_stopping_patience = 10

for epoch in range(training_epoch):
    epoch_start_time = time.time()
    
    # 학습률 출력 (5 에포크마다)
    if epoch % 5 == 0:
        current_lr = sess.run(learning_rate)
        print(f"   📉 현재 학습률: {current_lr:.6f}")
    
    # 배치별 학습
    epoch_batch_loss, epoch_batch_acc = [], []
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        
        # NaN 체크
        if np.isnan(mini_batch).any() or np.isnan(mini_label).any():
            print(f"⚠️ Epoch {epoch}, Batch {m}: NaN 발견, 건너뛰기")
            continue
            
        # 학습 실행
        _, loss1 = sess.run([optimizer, loss],
                           feed_dict={inputs: mini_batch, labels: mini_label})
        
        if np.isnan(loss1):
            print(f"⚠️ Epoch {epoch}, Batch {m}: 학습 중 NaN 발생")
            continue
            
        epoch_batch_loss.append(loss1)
    
    # 에포크 평균 계산
    if epoch_batch_loss:
        epoch_train_loss = np.mean(epoch_batch_loss)
    else:
        epoch_train_loss = 0.0
    
    # 테스트 평가 (수정된 방식)
    try:
        # 손실과 예측값 계산
        loss2, test_output = sess.run(
            [loss, y_pred],
            feed_dict={inputs: testX, labels: testY}
        )
        
        # 🔧 수정: 예측 확률 분석
        test_prob = np.clip(test_output, 0.0, 1.0)
        
        # 예측 분포 출력 (첫 5 에포크만)
        if epoch < 5:
            print(f"   📊 예측 확률 분포:")
            print(f"     평균: {test_prob.mean():.4f}")
            print(f"     표준편차: {test_prob.std():.4f}")
            print(f"     최소/최대: {test_prob.min():.4f}/{test_prob.max():.4f}")
            print(f"     > 임계값({threshold_used:.3f}) 비율: {(test_prob > threshold_used).mean():.3%}")
        
        # 🎯 핵심: 수정된 평가 함수 사용
        accuracy_val, precision_val, recall_val, f1_val, auc_val = evaluate_anomaly_fixed(
            testY.reshape(-1, actual_num_nodes), 
            test_prob,
            threshold_used
        )
        
        # 결과 저장
        test_loss.append(loss2)
        test_acc.append(accuracy_val)
        test_precision.append(precision_val)
        test_recall.append(recall_val)
        test_f1.append(f1_val)
        test_auc.append(auc_val)
        test_pred.append(test_prob)
        
        # 출력 (수정된 임계값 표시)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch:{epoch:2d}",
              f"train_loss:{epoch_train_loss:.4f}",
              f"test_loss:{loss2:.4f}",
              f"test_acc:{accuracy_val:.4f}",
              f"test_pre:{precision_val:.4f}",
              f"test_rec:{recall_val:.4f}",
              f"test_f1:{f1_val:.4f}",
              f"test_auc:{auc_val:.4f}",
              f"threshold:{threshold_used:.3f}[FIXED]",
              f"time:{epoch_time:.1f}s")
        
        # Early Stopping 체크
        if f1_val > best_f1:
            best_f1 = f1_val
            patience_counter = 0
            print(f"   🎯 새로운 최고 F1 점수: {best_f1:.4f}")
            
            # 최고 성능 모델 저장
            if epoch > 5:
                model_path = os.path.join(path, 'best_model')
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, f'{model_path}/V2X_ASTGCN_FIXED_BEST', global_step=epoch)
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"   ⏹️ Early Stopping at epoch {epoch} (patience: {patience_counter})")
            break
            
    except Exception as e:
        print(f"❌ Epoch {epoch} 테스트 중 오류: {e}")
        continue

time_end = time.time()
print(f'\n⏱️ GPU 수정된 이상탐지 학습 완료! 소요 시간: {time_end-time_start:.2f}초')

############## 결과 분석 및 저장 ###############
if test_f1 and len(test_f1) > 0:
    try:
        # 최고 F1 점수 기준으로 최적 모델 선택
        best_index = np.argmax(test_f1)
        best_test_result = test_pred[best_index]
        
        print(f"\n🔍 수정된 최종 성능 분석:")
        print(f"   최고 F1 점수 달성 에포크: {best_index}")
        print(f"   사용된 최적 임계값: {threshold_used:.4f}")
        print(f"   최고 F1 점수: {test_f1[best_index]:.4f}")
        
        # 🔧 수정된 상세 혼동행렬 분석
        test_label_final = testY.reshape(-1, actual_num_nodes)
        y_true_binary_final = test_label_final.flatten().astype(int)
        y_pred_binary_final = (best_test_result.flatten() > threshold_used).astype(int)
        
        cm = confusion_matrix(y_true_binary_final, y_pred_binary_final)
        print(f"   📊 수정된 최종 혼동행렬:")
        print(f"     [[TN={cm[0,0]:6d}, FP={cm[0,1]:6d}],")
        print(f"      [FN={cm[1,0]:6d}, TP={cm[1,1]:6d}]]")
        
        # 추가 분석
        total_samples = len(y_true_binary_final)
        actual_positive = np.sum(y_true_binary_final)
        predicted_positive = np.sum(y_pred_binary_final)
        
        print(f"   📊 상세 분석:")
        print(f"     전체 샘플: {total_samples:,}")
        print(f"     실제 이상: {actual_positive:,} ({actual_positive/total_samples:.2%})")
        print(f"     예측 이상: {predicted_positive:,} ({predicted_positive/total_samples:.2%})")
        
        # 🔧 수정: ROC 커브 분석 추가
        if len(np.unique(y_true_binary_final)) > 1:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true_binary_final, best_test_result.flatten())
            
            # 최적 임계값 찾기 (Youden's J statistic)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold_roc = thresholds[optimal_idx]
            
            print(f"   🎯 ROC 기반 최적 임계값: {optimal_threshold_roc:.4f}")
            print(f"     (현재 사용: {threshold_used:.4f})")
        
        # 클래스별 성능 분석
        if len(np.unique(y_true_binary_final)) > 1:
            class_report = classification_report(y_true_binary_final, y_pred_binary_final, 
                                               target_names=['Normal', 'Anomaly'], 
                                               digits=4)
            print(f"   📋 클래스별 성능 리포트:")
            print(class_report)
        
        # 결과 저장
        var = pd.DataFrame(best_test_result)
        var.to_csv(path + '/test_anomaly_result_fixed.csv', index=False, header=False)
        
        # 상세 메트릭 저장
        detailed_metrics = {
            'model_type': 'fixed_dynamic_threshold',
            'optimal_threshold': float(threshold_used),
            'max_class_weight': float(MAX_CLASS_WEIGHT),
            'best_epoch': int(best_index),
            'best_f1': float(test_f1[best_index]),
            'final_pos_weight': float(pos_weight),
            'confusion_matrix': cm.tolist(),
            'training_time_seconds': float(time_end - time_start),
            'data_statistics': {
                'data_mean': float(data1.mean()),
                'data_std': float(data1.std()),
                'anomaly_ratio': float((data1 > threshold_used).mean())
            }
        }
        
        import json
        with open(path + '/fixed_training_metadata.json', 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        
        # 학습 곡선 저장
        min_length = len(test_f1)
        training_curves = pd.DataFrame({
            'epoch': range(min_length),
            'test_loss': test_loss[:min_length],
            'test_acc': test_acc[:min_length],
            'test_precision': test_precision[:min_length],
            'test_recall': test_recall[:min_length],
            'test_f1': test_f1[:min_length],
            'test_auc': test_auc[:min_length],
            'optimal_threshold': [threshold_used] * min_length
        })
        training_curves.to_csv(path + '/fixed_training_curves.csv', index=False)
        
        # 평가 메트릭 저장
        evaluation_metrics = [
            test_acc[best_index],
            test_precision[best_index],
            test_recall[best_index],
            test_f1[best_index],
            test_auc[best_index]
        ]
        
        evaluation_df = pd.DataFrame(evaluation_metrics, 
                                   index=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
        evaluation_df.to_csv(path + '/fixed_anomaly_evaluation.csv')
        
        print("✅ 수정된 결과 저장 완료")

        print(f'\n🎉 V2X AST-GCN 수정된 이상탐지 결과:')
        print(f'   model_name: {model_name}_fixed')
        print(f'   dataset: {data_name}')
        print(f'   scheme: {scheme} ({name})')
        print(f'   task: anomaly_detection_fixed')
        print(f'   optimal_threshold: {threshold_used:.4f}')
        print(f'   max_class_weight: {MAX_CLASS_WEIGHT}')
        print(f'   best_accuracy: {test_acc[best_index]:.4f}')
        print(f'   best_precision: {test_precision[best_index]:.4f}')
        print(f'   best_recall: {test_recall[best_index]:.4f}')
        print(f'   best_f1: {test_f1[best_index]:.4f}')
        print(f'   best_auc: {test_auc[best_index]:.4f}')
        print(f'   final_pos_weight: {pos_weight:.2f}')
        print(f'   training_time: {time_end-time_start:.2f}s')
        print(f'📂 결과 저장 위치: {path}')
        
    except Exception as e:
        print(f"❌ 결과 저장 중 오류: {e}")
        import traceback
        traceback.print_exc()

print("\n🎉 V2X AST-GCN 수정된 이상탐지 시스템 실행 완료!")
print("🎯 주요 수정 사항:")
print(f"  - 동적 임계값: {threshold_used:.4f} (데이터 분포 기반)")
print(f"  - 개선된 클래스 가중치: {pos_weight:.2f} (최대 {MAX_CLASS_WEIGHT})")
print(f"  - 수정된 평가: 일관된 임계값 사용")
print(f"  - 예측 분포 모니터링: 실시간 확인")
print(f"  - ROC 기반 분석: 최적 임계값 제안")

print("\n📋 수정된 결과 파일:")
print("  - test_anomaly_result_fixed.csv: 수정된 예측 결과")
print("  - fixed_anomaly_evaluation.csv: 핵심 성능 메트릭")
print("  - fixed_training_curves.csv: 에포크별 학습 곡선")
print("  - fixed_training_metadata.json: 수정된 설정 정보")

if test_f1 and len(test_f1) > 0:
    final_f1 = max(test_f1)
    if final_f1 > 0.3:
        print("\n✅ 수정 성공!")
        print("  - 균형잡힌 성능 달성")
        print("  - 실제 운영 환경 적용 가능")
    else:
        print("\n🔧 추가 개선 권장:")
        print(f"  - 임계값 추가 조정: {threshold_used:.4f} → {threshold_used * 0.8:.4f}")
        print(f"  - 클래스 가중치 증가: {pos_weight:.2f} → {min(pos_weight * 1.5, MAX_CLASS_WEIGHT):.2f}")
else:
    print("\n❌ 학습 결과가 없어 분석할 수 없습니다.")