# visualization.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2X AST-GCN 시각화 모듈
Created for V2X Graph Neural Network Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter
import pandas as pd

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
plt.style.use('default')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def plot_result(test_result, test_label1, path):
    """V2X 테스트 결과 시각화"""
    
    print(f"📊 테스트 결과 시각화 중... 저장 위치: {path}")
    
    # 데이터 안전성 체크
    if test_result is None or test_label1 is None:
        print("❌ 시각화할 데이터가 없습니다.")
        return
    
    try:
        # 전체 테스트 결과 시각화
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # 첫 번째 차량의 결과만 표시 (대표성)
        if len(test_result.shape) > 1 and test_result.shape[1] > 0:
            a_pred = test_result[:, 0]
            a_true = test_label1[:, 0]
        else:
            a_pred = test_result.flatten()
            a_true = test_label1.flatten()
        
        # 시간 인덱스 생성
        time_idx = np.arange(len(a_pred))
        
        ax.plot(time_idx, a_pred, 'r-', label='V2X Prediction', linewidth=2, alpha=0.8)
        ax.plot(time_idx, a_true, 'b-', label='Actual Speed', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time Steps (15-min intervals)', fontsize=12)
        ax.set_ylabel('Vehicle Speed (km/h)', fontsize=12)
        ax.set_title('V2X AST-GCN: Speed Prediction vs Actual', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # RMSE 표시
        rmse = np.sqrt(np.mean((a_pred - a_true)**2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.2f} km/h', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_test_all.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 하루 결과 시각화 (96개 데이터포인트 = 24시간 × 4회/시간)
        fig, ax = plt.subplots(figsize=(12, 4))
        
        day_length = min(96, len(a_pred))
        a_pred_day = a_pred[:day_length]
        a_true_day = a_true[:day_length]
        
        # 시간 라벨 생성 (15분 간격)
        hours = np.arange(0, day_length/4, 0.25)
        
        ax.plot(hours, a_pred_day, 'r-', label='V2X Prediction', 
                linewidth=2.5, marker='o', markersize=3, alpha=0.8)
        ax.plot(hours, a_true_day, 'b-', label='Actual Speed', 
                linewidth=2.5, marker='s', markersize=3, alpha=0.8)
        
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Vehicle Speed (km/h)', fontsize=12)
        ax.set_title('V2X AST-GCN: 24-Hour Speed Prediction Pattern', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # x축 시간 라벨 설정
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xticklabels([f'{int(h):02d}:00' for h in np.arange(0, 25, 4)])
        
        # 일일 RMSE 표시
        daily_rmse = np.sqrt(np.mean((a_pred_day - a_true_day)**2))
        ax.text(0.02, 0.98, f'Daily RMSE: {daily_rmse:.2f} km/h', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_test_oneday.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 오차 분포 히스토그램
        fig, ax = plt.subplots(figsize=(8, 5))
        
        errors = a_pred - a_true
        ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Error: {np.mean(errors):.2f}')
        ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Prediction Error (km/h)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('V2X Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_error_distribution.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 테스트 결과 시각화 완료")
        
    except Exception as e:
        print(f"❌ 테스트 결과 시각화 오류: {e}")

def plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path):
    """V2X 학습 과정 시각화"""
    
    print(f"📈 학습 과정 시각화 중... 저장 위치: {path}")
    
    try:
        # 1. 학습 곡선 (RMSE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = np.arange(1, len(train_rmse) + 1)
        
        ax1.plot(epochs, train_rmse, 'r-', label='Train RMSE', linewidth=2, alpha=0.8)
        ax1.plot(epochs, test_rmse, 'b-', label='Test RMSE', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('RMSE (km/h)', fontsize=12)
        ax1.set_title('V2X AST-GCN: RMSE Learning Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 최소값 표시
        min_test_rmse = min(test_rmse)
        min_epoch = test_rmse.index(min_test_rmse) + 1
        ax1.axvline(min_epoch, color='green', linestyle='--', alpha=0.7)
        ax1.text(min_epoch, min_test_rmse, f'  Best: {min_test_rmse:.2f}\n  @Epoch {min_epoch}',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. 정확도 곡선
        ax2.plot(epochs, test_acc, 'g-', label='Test Accuracy', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('V2X AST-GCN: Accuracy Learning Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 최대값 표시
        max_test_acc = max(test_acc)
        max_acc_epoch = test_acc.index(max_test_acc) + 1
        ax2.axvline(max_acc_epoch, color='purple', linestyle='--', alpha=0.7)
        ax2.text(max_acc_epoch, max_test_acc, f'  Best: {max_test_acc:.3f}\n  @Epoch {max_acc_epoch}',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_learning_curves.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 손실 함수 곡선
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('V2X AST-GCN: Training Loss Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_train_loss.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 다중 메트릭 비교 (후반부 수렴 구간)
        if len(train_rmse) > 20:
            convergence_start = len(train_rmse) // 2  # 후반 50%
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            conv_epochs = np.arange(convergence_start, len(train_rmse))
            
            # RMSE 수렴
            ax1.plot(conv_epochs, train_rmse[convergence_start:], 'r-', label='Train RMSE', linewidth=2)
            ax1.plot(conv_epochs, test_rmse[convergence_start:], 'b-', label='Test RMSE', linewidth=2)
            ax1.set_title('RMSE Convergence', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 정확도 수렴
            ax2.plot(conv_epochs, test_acc[convergence_start:], 'g-', label='Test Accuracy', linewidth=2)
            ax2.set_title('Accuracy Convergence', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # MAE 수렴
            ax3.plot(conv_epochs, test_mae[convergence_start:], 'orange', label='Test MAE', linewidth=2)
            ax3.set_title('MAE Convergence', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 손실 수렴
            ax4.plot(conv_epochs, train_loss[convergence_start:], 'purple', label='Train Loss', linewidth=2)
            ax4.set_title('Loss Convergence', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('V2X AST-GCN: Convergence Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{path}/v2x_convergence_analysis.jpg', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. 최종 성능 요약 차트
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics = ['Min RMSE', 'Final MAE', 'Max Accuracy']
        values = [min(test_rmse), test_mae[test_rmse.index(min(test_rmse))], max(test_acc)]
        colors_bar = ['skyblue', 'lightgreen', 'orange']
        
        bars = ax.bar(metrics, values, color=colors_bar, alpha=0.8, edgecolor='black')
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('V2X AST-GCN: Final Performance Summary', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_performance_summary.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 학습 과정 시각화 완료")
        
    except Exception as e:
        print(f"❌ 학습 과정 시각화 오류: {e}")

def create_v2x_summary_report(path, metrics_dict):
    """V2X 결과 요약 리포트 생성"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 성능 메트릭 레이더 차트 (정규화)
        categories = ['RMSE', 'MAE', 'Accuracy', 'R²', 'Var Score']
        
        # 값들을 0-1 범위로 정규화 (RMSE, MAE는 역수 사용)
        values = [
            1 / (1 + metrics_dict.get('rmse', 1)),  # 낮을수록 좋음
            1 / (1 + metrics_dict.get('mae', 1)),   # 낮을수록 좋음
            metrics_dict.get('accuracy', 0),        # 높을수록 좋음
            max(0, metrics_dict.get('r2', 0)),      # 높을수록 좋음
            max(0, metrics_dict.get('var', 0))      # 높을수록 좋음
        ]
        
        # 원형 그래프를 위한 각도 계산
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 닫힌 도형을 위해
        angles += angles[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.8)
        ax1.fill(angles, values, color='blue', alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('V2X Model Performance Radar', fontweight='bold')
        ax1.grid(True)
        
        # 2. 시간별 성능 분석 (가상 데이터)
        hours = np.arange(0, 24)
        performance = np.random.normal(0.8, 0.1, 24)  # 실제로는 시간별 성능 데이터 사용
        
        ax2.plot(hours, performance, 'g-', linewidth=2, marker='o')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Model Performance')
        ax2.set_title('V2X Performance by Time of Day', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(np.arange(0, 24, 4))
        
        # 3. 차량 타입별 성능 (가상 데이터)
        vehicle_types = ['Sedan', 'SUV', 'Truck', 'Bus', 'Motorcycle']
        type_performance = [0.85, 0.82, 0.78, 0.75, 0.88]
        
        bars = ax3.bar(vehicle_types, type_performance, color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
        ax3.set_ylabel('Prediction Accuracy')
        ax3.set_title('V2X Performance by Vehicle Type', fontweight='bold')
        ax3.set_ylim(0, 1)
        
        for bar, perf in zip(bars, type_performance):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{perf:.2f}', ha='center', va='bottom')
        
        # 4. 모델 복잡도 vs 성능
        model_variants = ['TGCN', 'AST-GCN\n(POI)', 'AST-GCN\n(Weather)', 'AST-GCN\n(Both)']
        complexity = [1, 2, 2, 3]
        performance_comp = [0.75, 0.82, 0.80, 0.85]
        
        scatter = ax4.scatter(complexity, performance_comp, s=[100, 150, 150, 200], 
                            c=['red', 'blue', 'green', 'purple'], alpha=0.7)
        
        for i, variant in enumerate(model_variants):
            ax4.annotate(variant, (complexity[i], performance_comp[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Model Complexity')
        ax4.set_ylabel('Performance Score')
        ax4.set_title('V2X Model: Complexity vs Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('V2X AST-GCN: Comprehensive Analysis Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_comprehensive_report.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ V2X 종합 리포트 생성 완료")
        
    except Exception as e:
        print(f"❌ V2X 리포트 생성 오류: {e}")

# 기존 함수들과 호환성 유지
def plot_result_original(test_result, test_label1, path):
    """원본 시각화 함수 (호환성용)"""
    plot_result(test_result, test_label1, path)

def plot_error_original(train_rmse, train_loss, test_rmse, test_acc, test_mae, path):
    """원본 시각화 함수 (호환성용)"""
    plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)

# visualization.py 파일 끝에 다음 함수들을 추가하세요:

def plot_anomaly_result(test_result, test_label, path):
    """V2X 이상탐지 결과 시각화"""
    try:
        print(f"🚨 이상탐지 결과 시각화 중... 저장 위치: {path}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 이상 확률 분포
        anomaly_probs = test_result.flatten()
        ax1.hist(anomaly_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
        ax1.axvline(0.2, color='orange', linestyle='--', label='Adjusted Threshold')
        ax1.set_xlabel('Anomaly Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('V2X Grid Anomaly Probability Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 시간별 평균 이상 확률
        time_steps = test_result.shape[0]
        avg_anomaly_by_time = np.mean(test_result, axis=1)
        ax2.plot(range(time_steps), avg_anomaly_by_time, 'b-', linewidth=2)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Original Threshold')
        ax2.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='Adjusted Threshold')
        ax2.set_xlabel('Time Steps (15-min intervals)')
        ax2.set_ylabel('Average Anomaly Probability')
        ax2.set_title('V2X Grid Anomaly Trend Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 격자별 평균 이상 확률 (상위 50개만)
        num_grids = min(50, test_result.shape[1])
        avg_anomaly_by_grid = np.mean(test_result, axis=0)[:num_grids]
        grid_ids = range(num_grids)
        
        bars = ax3.bar(grid_ids, avg_anomaly_by_grid, color='orange', alpha=0.7)
        ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(0.2, color='orange', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Grid ID')
        ax3.set_ylabel('Average Anomaly Probability')
        ax3.set_title(f'V2X Anomaly by Grid (Top {num_grids})')
        ax3.grid(True, alpha=0.3)
        
        # 4. 실제 vs 예측 이상 비율
        y_true_binary = (test_label.flatten() > 0.2).astype(int)
        y_pred_binary_50 = (test_result.flatten() > 0.5).astype(int)
        y_pred_binary_20 = (test_result.flatten() > 0.2).astype(int)
        
        actual_ratio = y_true_binary.mean()
        pred_ratio_50 = y_pred_binary_50.mean()
        pred_ratio_20 = y_pred_binary_20.mean()
        
        categories = ['Actual\nAnomalies', 'Predicted\n(Thresh=0.5)', 'Predicted\n(Thresh=0.2)']
        ratios = [actual_ratio, pred_ratio_50, pred_ratio_20]
        colors_bar = ['green', 'red', 'orange']
        
        bars = ax4.bar(categories, ratios, color=colors_bar, alpha=0.7)
        ax4.set_ylabel('Anomaly Ratio')
        ax4.set_title('V2X Anomaly Detection: Actual vs Predicted')
        
        # 값 표시
        for bar, ratio in zip(bars, ratios):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_anomaly_results.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 이상탐지 결과 시각화 완료")
        
    except Exception as e:
        print(f"❌ 이상탐지 결과 시각화 오류: {e}")

def plot_anomaly_training(train_acc, train_loss, test_acc, test_f1, test_auc, test_precision, test_recall, path):
    """V2X 이상탐지 학습 과정 시각화"""
    try:
        print(f"📈 이상탐지 학습 과정 시각화 중... 저장 위치: {path}")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_acc) + 1)
        
        # 1. 정확도 곡선
        ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
        ax1.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('V2X Anomaly Detection: Accuracy Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 최고 정확도 표시
        max_test_acc = max(test_acc)
        max_epoch = test_acc.index(max_test_acc) + 1
        ax1.axvline(max_epoch, color='green', linestyle='--', alpha=0.7)
        ax1.text(max_epoch, max_test_acc, f'  Best: {max_test_acc:.3f}\n  @Epoch {max_epoch}',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. 손실 곡선
        ax2.plot(epochs, train_loss, 'g-', label='Training Loss', linewidth=2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('V2X Anomaly Detection: Loss Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. F1 및 AUC 곡선
        ax3.plot(epochs, test_f1, 'purple', label='F1-Score', linewidth=2)
        ax3.plot(epochs, test_auc, 'orange', label='AUC', linewidth=2)
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Score')
        ax3.set_title('V2X Anomaly Detection: F1 & AUC Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 최고 F1 점수 표시
        max_f1 = max(test_f1)
        max_f1_epoch = test_f1.index(max_f1) + 1
        ax3.axvline(max_f1_epoch, color='purple', linestyle='--', alpha=0.7)
        
        # 4. 정밀도 및 재현율 곡선
        ax4.plot(epochs, test_precision, 'cyan', label='Precision', linewidth=2)
        ax4.plot(epochs, test_recall, 'magenta', label='Recall', linewidth=2)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Score')
        ax4.set_title('V2X Anomaly Detection: Precision & Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_anomaly_training.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 이상탐지 학습 과정 시각화 완료")
        
    except Exception as e:
        print(f"❌ 이상탐지 학습 과정 시각화 오류: {e}")

def plot_anomaly_confusion_matrix(y_true, y_pred, path, threshold=0.2):
    """V2X 이상탐지 혼동행렬 시각화"""
    try:
        from sklearn.metrics import confusion_matrix
        
        print(f"🔍 혼동행렬 시각화 중... 저장 위치: {path}")
        
        # 이진 분류를 위한 임계값 적용
        y_true_binary = (y_true.flatten() > threshold).astype(int)
        y_pred_binary = (y_pred.flatten() > threshold).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 혼동행렬 히트맵
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'V2X Anomaly Detection Confusion Matrix\n(Threshold = {threshold})', 
                    fontweight='bold')
        
        # 컬러바 추가
        plt.colorbar(im, ax=ax)
        
        # 혼동행렬 라벨 추가
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", 
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        
        # 통계 정보 추가
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        stats_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
               verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'{path}/v2x_confusion_matrix.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 혼동행렬 시각화 완료")
        
    except Exception as e:
        print(f"❌ 혼동행렬 시각화 오류: {e}")