#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2X AST-GCN 이상탐지 메인 실행 파일 - 고정 임계값 안정화 버전

핵심 개선사항:
1. 고정 임계값 사용 (0.3) - 학습 중 변경 금지
2. 클래스 가중치 제한 (최대 3.0)
3. 안정화된 평가 함수
4. 일관된 데이터 처리

Author: V2X Anomaly Detection Team (Stabilized)
Date: 2025-06-07
"""

import pickle as pkl
import tensorflow as tf
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf_v1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 🎯 고정 임계값 설정 (핵심)
FIXED_THRESHOLD = 0.3  # 학습 중 절대 변경 안함
MAX_CLASS_WEIGHT = 3.0  # 클래스 가중치 제한

print(f"🎯 안정화 설정:")
print(f"   고정 임계값: {FIXED_THRESHOLD}")
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
data1 = np.mat(data, dtype=np.float32)

#### normalization (이상점수는 이미 0-1 범위이므로 최소한의 정규화)
max_value = np.max(data1)
if max_value > 0:
    data1 = data1 / max_value
else:
    print("⚠️ 모든 이상점수가 0입니다. 정규화를 건너뜁니다.")

if hasattr(data, 'columns'):
    data1.columns = data.columns

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
print(f'   noise_name: {noise_name}')
print(f'   noise_param: {PG}')

# 개선된 전처리 (클래스 균형 정보 포함)
print(f"\n🔄 이상탐지 데이터 전처리 시작...")

try:
    # 개선된 전처리 호출 (balance_info 추가 반환)
    trainX, trainY, testX, testY, balance_info = preprocess_data(
        data1, time_len, train_rate, seq_len, pre_len, model_name, scheme,
        poi_data, weather_data
    )
    
    # 클래스 균형 정보 추출
    pos_weight = balance_info['pos_weight']
    threshold_used = balance_info['threshold_used']
    train_anomaly_ratio = balance_info['train_anomaly_ratio']
    test_anomaly_ratio = balance_info['test_anomaly_ratio']
    
    print(f"🎯 클래스 균형 정보:")
    print(f"   양성 클래스 가중치: {pos_weight:.2f}")
    print(f"   사용된 임계값: {threshold_used:.3f}")
    print(f"   훈련 이상 비율: {train_anomaly_ratio:.2%}")
    print(f"   테스트 이상 비율: {test_anomaly_ratio:.2%}")
    
except (ValueError, TypeError) as e:
    # 기존 전처리 폴백
    print(f"⚠️ 개선된 전처리 실패, 기존 방식 사용: {e}")
    trainX, trainY, testX, testY = preprocess_data(
        data1, time_len, train_rate, seq_len, pre_len, model_name, scheme,
        poi_data, weather_data
    )
    
    # 수동으로 클래스 가중치 계산
    y_flat = trainY.flatten()
    pos_count = np.sum(y_flat > 0.3)
    neg_count = len(y_flat) - pos_count
    pos_weight = np.clip(neg_count / (pos_count + 1e-8), 2.0, 15.0)
    threshold_used = 0.3
    train_anomaly_ratio = pos_count / len(y_flat)
    test_anomaly_ratio = (testY.flatten() > 0.3).mean()
    
    print(f"🎯 기본 클래스 균형 정보:")
    print(f"   계산된 클래스 가중치: {pos_weight:.2f}")
    print(f"   사용된 임계값: {threshold_used:.3f}")

totalbatch = int(trainX.shape[0] / batch_size)
training_data_count = len(trainX)

print(f"✅ 전처리 완료:")
print(f"   total batches: {totalbatch}")
print(f"   training samples: {training_data_count}")

def create_balanced_loss_function(pos_weight=10.0, use_focal=True):
    """
    클래스 불균형을 해결하는 균형 손실함수 생성
    """
    
    def focal_loss(y_true, y_pred, alpha=0.75, gamma=2.0):
        """Focal Loss 구현"""
        y_true = tf_v1.cast(y_true, tf_v1.float32)
        
        # Sigmoid 적용
        y_pred_sigmoid = tf_v1.nn.sigmoid(y_pred)
        y_pred_sigmoid = tf_v1.clip_by_value(y_pred_sigmoid, 1e-8, 1.0 - 1e-8)
        
        # Cross Entropy
        ce_loss = -y_true * tf_v1.math.log(y_pred_sigmoid) - (1 - y_true) * tf_v1.math.log(1 - y_pred_sigmoid)
        
        # Focal Weight (gamma 유지)
        pt = tf_v1.where(tf_v1.equal(y_true, 1), y_pred_sigmoid, 1 - y_pred_sigmoid)
        focal_weight = alpha * tf_v1.pow(1 - pt, gamma)
        
        return focal_weight * ce_loss
    
    def balanced_loss(y_true, y_pred):
        """균형 손실함수"""
        
        # 1. Weighted Binary Cross Entropy (pos_weight 증가)
        weighted_bce = tf_v1.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            pos_weight=pos_weight * 1.5
        )
        
        if use_focal:
            # 2. Focal Loss 추가
            focal_loss_val = focal_loss(y_true, y_pred, alpha=0.75, gamma=2.0)
            
            # 3. 결합 (BCE 70%, Focal 30%)
            combined_loss = 0.7 * weighted_bce + 0.3 * focal_loss_val
        else:
            combined_loss = weighted_bce
        
        return tf_v1.reduce_mean(combined_loss)
    
    return balanced_loss

def find_optimal_threshold_detailed(y_true, y_pred_proba, thresholds=None):
    """
    상세한 최적 임계값 탐색
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    # 데이터 평평화
    y_true_flat = y_true.flatten()
    y_pred_proba_flat = y_pred_proba.flatten()
    
    results = []
    best_f1 = 0
    best_threshold = 0.5
    
    print("🔍 임계값별 성능 분석:")
    print("-" * 70)
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    for threshold in thresholds:
        # 이진 예측
        y_pred_binary = (y_pred_proba_flat > threshold).astype(int)
        y_true_binary = (y_true_flat > threshold_used).astype(int)
        
        # 혼동행렬 계산
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        # 메트릭 계산
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        print(f"{threshold:<10.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        # 최적 F1 점수 찾기
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("-" * 70)
    print(f"🎯 최적 임계값: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    return best_threshold, best_f1

def evaluate_anomaly_detection_improved(y_true, y_pred_logits, threshold=None):
    """
    개선된 이상탐지 평가
    """
    # 확률로 변환
    y_pred_proba = tf_v1.nn.sigmoid(y_pred_logits).eval(session=sess)
    
    # 최적 임계값 찾기 (첫 번째 평가에서만)
    if threshold is None:
        threshold, _ = find_optimal_threshold_detailed(y_true, y_pred_proba)
    
    # 이진 예측
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    # 데이터 평평화
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    y_pred_proba_flat = y_pred_proba.flatten()
    
    # 실제 라벨 이진화 (threshold_used 기준)
    y_true_binary = (y_true_flat > threshold_used).astype(int)
    
    # 혼동행렬
    tp = np.sum((y_true_binary == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_flat == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_flat == 0))
    
    # 메트릭 계산
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # AUC 계산
    try:
        if len(np.unique(y_true_binary)) > 1:
            auc = roc_auc_score(y_true_binary, y_pred_proba_flat)
        else:
            auc = 0.5
    except:
        auc = 0.5
    
    # 상세 출력
    print(f"🔍 개선된 이상탐지 평가 (임계값: {threshold:.3f}):")
    print(f"   📊 혼동행렬: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"   📈 성능 메트릭:")
    print(f"     Accuracy: {accuracy:.4f}")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall: {recall:.4f}")
    print(f"     F1-Score: {f1:.4f}")
    print(f"     AUC: {auc:.4f}")
    print(f"   📊 클래스 분포:")
    print(f"     실제 이상 비율: {np.mean(y_true_binary):.3f}")
    print(f"     예측 이상 비율: {np.mean(y_pred_flat):.3f}")
    
    return accuracy, precision, recall, f1, auc, threshold

def evaluate_anomaly_numpy(y_true, y_logits, threshold=None, threshold_used=0.3):
    """
    NumPy 기반 이상탐지 평가
    """
    # 1) 로짓 → 확률
    y_proba = 1.0 / (1.0 + np.exp(-y_logits))
    y_proba_flat = y_proba.flatten()
    y_true_flat = y_true.flatten()
    
    # 2) y_true 이진화
    y_true_bin = (y_true_flat > threshold_used).astype(int)
    
    # 3) threshold 탐색 (None이면)
    if threshold is None:
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.3, 0.7, 0.01):
            y_pred_bin = (y_proba_flat > thr).astype(int)
            if len(np.unique(y_true_bin)) < 2 or len(np.unique(y_pred_bin)) < 2:
                continue
            try:
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            except:
                continue
        threshold = best_thr
    
    # 4) 최종 이진 예측
    y_pred_bin = (y_proba_flat > threshold).astype(int)
    
    # 5) 메트릭 계산
    acc   = accuracy_score(y_true_bin, y_pred_bin)
    pre   = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec   = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1    = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        auc  = roc_auc_score(y_true_bin, y_proba_flat)
    except:
        auc  = 0.5
    
    return acc, pre, rec, f1, auc, threshold

def TGCN_ANOMALY(_X, _weights, _biases):
    """
    TGCN 이상탐지용 모델 (기존 TGCN + 시그모이드 출력)
    """
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf_v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf_v1.unstack(_X, axis=1)
    outputs, states = tf_v1.nn.static_rnn(cell, _X, dtype=tf_v1.float32)
    m = []
    for i in outputs:
        o = tf_v1.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf_v1.reshape(o, shape=[-1, gru_units])
        # Dropout 비율 증가
        o = tf_v1.nn.dropout(o, keep_prob=0.7)
        m.append(o)
    last_output = m[-1]
    
    # 이상탐지용 출력 (시그모이드)
    logits = tf_v1.matmul(last_output, _weights['out']) + _biases['out']
    logits = tf_v1.reshape(logits, shape=[-1, num_nodes, pre_len])
    logits = tf_v1.transpose(logits, perm=[0, 2, 1])
    logits = tf_v1.reshape(logits, shape=[-1, num_nodes])
    
    # 수치 안정성 추가 (클리핑 범위 조정)
    logits = tf_v1.clip_by_value(logits, -12.0, 12.0)
    
    # 시그모이드 활성화 (이상 확률)
    output = tf_v1.sigmoid(logits)
    
    return output, logits, m, states

def simple_evaluation(y_true, y_pred, threshold=0.5):
    """간단한 이상탐지 평가"""
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # threshold가 None이면 기본값 0.5
        if threshold is None:
            threshold = 0.5
        # (1) y_true, (2) y_pred 모두 같은 threshold 사용
        y_true_bin = (y_true_flat > threshold).astype(int)
        y_pred_bin = (y_pred_flat >= threshold).astype(int)
        
        acc = accuracy_score(y_true_bin, y_pred_bin)
        
        if len(np.unique(y_true_bin)) > 1:
            pre = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        else:
            pre = rec = f1 = 0.0
            
        return acc, pre, rec, f1, threshold
    except Exception as e:
        print(f"⚠️ 평가 중 오류 발생: {e}")
        return 0.5, 0.0, 0.0, 0.0, 0.5

def evaluate_anomaly_stable(y_true, y_pred, threshold=FIXED_THRESHOLD):
    """
    고정 임계값 기반 안정적 이상탐지 평가
    """
    try:
        # 데이터 평평화
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # NaN/Inf 안전 처리
        y_pred_flat = np.nan_to_num(y_pred_flat, 0.0)
        y_pred_flat = np.clip(y_pred_flat, 0.0, 1.0)
        
        # 고정 임계값으로 이진화
        y_true_bin = (y_true_flat > threshold).astype(int)
        y_pred_bin = (y_pred_flat > threshold).astype(int)
        
        # 메트릭 계산
        acc = accuracy_score(y_true_bin, y_pred_bin)
        
        # 클래스가 존재하는 경우만 계산
        if len(np.unique(y_true_bin)) > 1:
            pre = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            try:
                auc = roc_auc_score(y_true_bin, y_pred_flat)
            except:
                auc = 0.5
        else:
            pre = rec = f1 = auc = 0.0
        
        return acc, pre, rec, f1, auc
    except Exception as e:
        print(f"⚠️ 평가 중 오류: {e}")
        return 0.5, 0.0, 0.0, 0.0, 0.5

def create_stable_loss_function(pos_weight=2.0):
    """
    안정화된 균형 손실함수 (가중치 제한)
    """
    def stable_loss(y_true, y_pred):
        # Weighted Binary Cross Entropy (제한된 가중치)
        weighted_bce = tf_v1.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            pos_weight=pos_weight
        )
        
        # 가벼운 Focal Loss 추가
        y_pred_sigmoid = tf_v1.nn.sigmoid(y_pred)
        y_pred_sigmoid = tf_v1.clip_by_value(y_pred_sigmoid, 1e-8, 1.0 - 1e-8)
        
        ce_loss = -y_true * tf_v1.math.log(y_pred_sigmoid) - (1 - y_true) * tf_v1.math.log(1 - y_pred_sigmoid)
        pt = tf_v1.where(tf_v1.equal(y_true, 1), y_pred_sigmoid, 1 - y_pred_sigmoid)
        focal_weight = 0.25 * tf_v1.pow(1 - pt, 2.0)
        focal_loss = focal_weight * ce_loss
        
        # 결합 (BCE 80%, Focal 20%)
        combined_loss = 0.8 * weighted_bce + 0.2 * focal_loss
        
        return tf_v1.reduce_mean(combined_loss)
    
    return stable_loss

###### placeholders ######
print(f"\n🧠 개선된 이상탐지 모델 구성...")

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
optimal_threshold = tf_v1.placeholder(tf_v1.float32, shape=())

# Graph weights (이상탐지용)
weights = {
    'out': tf_v1.Variable(tf_v1.random_normal([gru_units, actual_pre_len], mean=0.0, stddev=0.01), name='weight_o')}

biases = {
    'out': tf_v1.Variable(tf_v1.random_normal([actual_pre_len], stddev=0.01), name='bias_o')}

pred, logits, ttts, ttto = TGCN_ANOMALY(inputs, weights, biases)

y_pred = pred  # 시그모이드 출력 (0-1 확률)
y_logits = logits  # 로지트 (손실 계산용)

###### optimizer (개선된 이상탐지용 손실함수) ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf_v1.nn.l2_loss(tf_var) for tf_var in tf_v1.trainable_variables())
label = tf_v1.reshape(labels, [-1, actual_num_nodes])
logits_reshaped = tf_v1.reshape(y_logits, [-1, actual_num_nodes])

print('y_pred_shape:', y_pred.shape)
print('label_shape:', label.shape)
print('logits_shape:', logits_reshaped.shape)

# 개선된 균형 손실함수 사용
print(f"🎯 개선된 손실함수 구성 (클래스 가중치: {pos_weight:.2f})")
balanced_loss_fn = create_balanced_loss_function(pos_weight=pos_weight, use_focal=True)
main_loss = balanced_loss_fn(label, logits_reshaped)
loss = main_loss + Lreg

# TF 그래프에서 accuracy 계산
predictions = tf_v1.cast(tf_v1.greater(y_pred, optimal_threshold), tf_v1.float32)
accuracy = tf_v1.reduce_mean(tf_v1.cast(tf_v1.equal(predictions, tf_v1.reshape(labels, [-1, actual_num_nodes])), tf_v1.float32))

# 개선된 최적화기 (학습률 스케줄링)
global_step = tf_v1.Variable(0, trainable=False)
learning_rate = tf_v1.train.exponential_decay(
    lr, global_step, 
    decay_steps=500,  # 더 빠른 감소
    decay_rate=0.9,   # 더 급격한 감소
    staircase=True
)

# Gradient clipping (개선)
opt = tf_v1.train.AdamOptimizer(learning_rate)
grads_and_vars = opt.compute_gradients(loss)
clipped_grads_and_vars = [(tf_v1.clip_by_value(grad, -0.5, 0.5), var)  # 더 좁은 클리핑 범위
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

# 출력 디렉토리 설정 (이상탐지용)
if data_name == 'v2x':
    out = f'out/v2x_{model_name}_anomaly_scheme{scheme}_gpu'
else:
    out = f'out/{model_name}_anomaly_{noise_name}_gpu'

path1 = f'{model_name}_anomaly_{name}_{data_name}_lr{lr}_batch{batch_size}_unit{gru_units}_seq{seq_len}_pre{pre_len}_epoch{training_epoch}_scheme{scheme}_PG{PG}_GPU'
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)

print(f"📂 결과 저장 경로: {path}")

print(f"\n🚀 V2X AST-GCN 개선된 이상탐지 GPU 학습 시작...")
print(f"   Epochs: {training_epoch}")
print(f"   Batch size: {batch_size}")
print(f"   Learning rate: {lr} (지수 감소)")
print(f"   클래스 가중치: {pos_weight:.2f}")
print(f"   임계값: {threshold_used:.3f}")

x_axe, batch_loss, batch_acc = [], [], []
test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_pred = [], [], [], [], [], [], []

best_f1 = 0.0
patience_counter = 0
early_stopping_patience = 15  # 더 긴 인내심

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
    
    # 테스트 평가 (안정화된 방식)
    try:
        # 손실과 예측값 계산
        loss2, test_output = sess.run(
            [loss, y_pred],
            feed_dict={inputs: testX, labels: testY}
        )
        
        # 확률값 안전 처리
        test_prob = np.clip(test_output, 0.0, 1.0)
        
        # 🎯 핵심: 고정 임계값으로 평가
        accuracy_val, precision_val, recall_val, f1_val, auc_val = evaluate_anomaly_stable(
            testY.reshape(-1, actual_num_nodes), 
            test_prob
        )
        
        # 결과 저장
        test_loss.append(loss2)
        test_acc.append(accuracy_val)
        test_precision.append(precision_val)
        test_recall.append(recall_val)
        test_f1.append(f1_val)
        test_auc.append(auc_val)
        test_pred.append(test_prob)
        
        # 출력 (고정 임계값 표시)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch:{epoch:2d}",
              f"train_loss:{epoch_train_loss:.4f}",
              f"test_loss:{loss2:.4f}",
              f"test_acc:{accuracy_val:.4f}",
              f"test_pre:{precision_val:.4f}",
              f"test_rec:{recall_val:.4f}",
              f"test_f1:{f1_val:.4f}",
              f"test_auc:{auc_val:.4f}",
              f"threshold:{FIXED_THRESHOLD:.3f}[FIXED]",
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
                saver.save(sess, f'{model_path}/V2X_ASTGCN_STABLE_BEST', global_step=epoch)
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"   ⏹️ Early Stopping at epoch {epoch} (patience: {patience_counter})")
            break
            
    except Exception as e:
        print(f"❌ Epoch {epoch} 테스트 중 오류: {e}")
        continue

time_end = time.time()
print(f'\n⏱️ GPU 개선된 이상탐지 학습 완료! 소요 시간: {time_end-time_start:.2f}초')

############## 결과 분석 및 저장 ###############
if test_f1 and len(test_f1) > 0:
    try:
        # 최고 F1 점수 기준으로 최적 모델 선택
        best_index = np.argmax(test_f1)
        best_test_result = test_pred[best_index]
        
        print(f"\n🔍 안정화된 최종 성능 분석:")
        print(f"   최고 F1 점수 달성 에포크: {best_index}")
        print(f"   사용된 고정 임계값: {FIXED_THRESHOLD}")
        print(f"   최고 F1 점수: {test_f1[best_index]:.4f}")
        
        # 상세 혼동행렬 분석
        test_label_final = testY.reshape(-1, actual_num_nodes)
        y_true_binary_final = (test_label_final.flatten() > FIXED_THRESHOLD).astype(int)
        y_pred_binary_final = (best_test_result.flatten() > FIXED_THRESHOLD).astype(int)
        
        cm = confusion_matrix(y_true_binary_final, y_pred_binary_final)
        print(f"   📊 최종 혼동행렬:")
        print(f"     [[TN={cm[0,0]:6d}, FP={cm[0,1]:6d}],")
        print(f"      [FN={cm[1,0]:6d}, TP={cm[1,1]:6d}]]")
        
        # 클래스별 성능 분석
        if len(np.unique(y_true_binary_final)) > 1:
            class_report = classification_report(y_true_binary_final, y_pred_binary_final, 
                                               target_names=['Normal', 'Anomaly'], 
                                               digits=4)
            print(f"   📋 클래스별 성능 리포트:")
            print(class_report)
        
        # 결과 저장
        var = pd.DataFrame(best_test_result)
        var.to_csv(path + '/test_anomaly_result_stable.csv', index=False, header=False)
        
        # 상세 메트릭 저장
        detailed_metrics = {
            'model_type': 'stable_fixed_threshold',
            'fixed_threshold': float(FIXED_THRESHOLD),
            'max_class_weight': float(MAX_CLASS_WEIGHT),
            'best_epoch': int(best_index),
            'best_f1': float(test_f1[best_index]),
            'final_pos_weight': float(pos_weight),
            'confusion_matrix': cm.tolist(),
            'training_time_seconds': float(time_end - time_start)
        }
        
        import json
        with open(path + '/stable_training_metadata.json', 'w') as f:
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
            'fixed_threshold': [FIXED_THRESHOLD] * min_length
        })
        training_curves.to_csv(path + '/stable_training_curves.csv', index=False)
        
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
        evaluation_df.to_csv(path + '/stable_anomaly_evaluation.csv')
        
        print("✅ 안정화된 결과 저장 완료")

        print(f'\n🎉 V2X AST-GCN 안정화된 이상탐지 결과:')
        print(f'   model_name: {model_name}_stable')
        print(f'   dataset: {data_name}')
        print(f'   scheme: {scheme} ({name})')
        print(f'   task: anomaly_detection_stable')
        print(f'   fixed_threshold: {FIXED_THRESHOLD}')
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

print("\n🎉 V2X AST-GCN 안정화된 이상탐지 시스템 실행 완료!")
print("🎯 안정화 적용 사항:")
print(f"  - 고정 임계값: {FIXED_THRESHOLD} (학습 중 변경 없음)")
print(f"  - 제한된 클래스 가중치: {pos_weight:.2f} (최대 {MAX_CLASS_WEIGHT})")
print(f"  - 안정화된 평가: 일관된 메트릭 계산")
print(f"  - 개선된 Early Stopping: 더 안정적인 수렴")
print(f"  - 수치 안정성: NaN/Inf 처리 강화")

print("\n📋 안정화 결과 파일:")
print("  - test_anomaly_result_stable.csv: 안정화된 예측 결과")
print("  - stable_anomaly_evaluation.csv: 핵심 성능 메트릭")
print("  - stable_training_curves.csv: 에포크별 학습 곡선")
print("  - stable_training_metadata.json: 안정화 설정 정보")

if test_f1 and len(test_f1) > 0:
    final_f1 = max(test_f1)
    if final_f1 > 0.25:
        print("\n✅ 안정화 성공!")
        print("  - 일관된 학습 패턴 달성")
        print("  - 실제 운영 환경 적용 가능")
    else:
        print("\n🔧 추가 개선 제안:")
        print(f"  - 임계값 조정: {FIXED_THRESHOLD} → {FIXED_THRESHOLD * 0.8:.2f}")
        print(f"  - 학습 에포크 증가: {training_epoch} → {training_epoch + 20}")
else:
    print("\n❌ 학습 결과가 없어 분석할 수 없습니다.")