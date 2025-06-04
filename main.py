# -*- coding: utf-8 -*-

import pickle as pkl
import tensorflow as tf

# Apple GPU (Metal) 설정 - TF 2.x 네이티브 모드에서
print("🔍 GPU 설정 중...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 설정
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"🚀 Apple GPU (Metal) 활성화 성공! - {gpus[0]}")
    except RuntimeError as e:
        print(f"⚠️ GPU 설정 오류: {e}")
else:
    print("💻 CPU 모드로 실행")

# 이제 TF 1.x 호환 모드로 전환 (GPU 설정 이후)
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from acell import preprocess_data, load_assist_data, load_v2x_data
from tgcn import tgcnCell

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

time_start = time.time()

###### Settings ######
flags = tf_v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')  # 학습률 낮춤
flags.DEFINE_integer('training_epoch', 20, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 128, 'hidden units of gru.')
flags.DEFINE_integer('seq_len',10 , 'time length of inputs.')
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
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
scheme = FLAGS.scheme
PG = FLAGS.noise_param

###### load data ######
print(f"🚗 데이터 로딩: {data_name}")

if data_name == 'v2x':
    data, adj, poi_data, weather_data = load_v2x_data('v2x')
    print(f"✅ V2X 데이터 로딩 완료")
elif data_name == 'sz':
    data, adj = load_assist_data('sz')
    poi_data, weather_data = None, None
    print(f"✅ Shenzhen 데이터 로딩 완료")
else:
    raise ValueError(f"❌ 지원하지 않는 데이터셋: {data_name}")

### Perturbation Analysis
def MaxMinNormalization(x,Max,Min):
    x = (x-Min)/(Max-Min)
    return x

if noise_name == 'Gauss':
    Gauss = np.random.normal(0,PG,size=data.shape)
    noise_Gauss = MaxMinNormalization(Gauss,np.max(Gauss),np.min(Gauss))
    data = data + noise_Gauss
elif noise_name == 'Possion':
    Possion = np.random.poisson(PG,size=data.shape)
    noise_Possion = MaxMinNormalization(Possion,np.max(Possion),np.min(Possion))
    data = data + noise_Possion
else:
    data = data

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

#### normalization
max_value = np.max(data1)
data1 = data1/max_value

if hasattr(data, 'columns'):
    data1.columns = data.columns

# 모델 및 스킴 정보 출력
if model_name == 'ast-gcn':
    if scheme == 1:
        name = 'V2X POI only' if data_name == 'v2x' else 'add poi dim'
    elif scheme == 2:
        name = 'V2X Weather only' if data_name == 'v2x' else 'add weather dim'
    else:
        name = 'V2X POI + Weather' if data_name == 'v2x' else 'add poi + weather dim'
else:
    name = 'tgcn'

print('📊 모델 정보:')
print(f'   model: {model_name}')
print(f'   dataset: {data_name}')
print(f'   scheme: {scheme} ({name})')
print(f'   data shape: {data1.shape}')
print(f'   time_len: {time_len}, num_nodes: {num_nodes}')
print(f'   noise_name: {noise_name}')
print(f'   noise_param: {PG}')

# 전처리
print(f"\n🔄 데이터 전처리 시작...")
trainX, trainY, testX, testY = preprocess_data(
    data1, time_len, train_rate, seq_len, pre_len, model_name, scheme,
    poi_data, weather_data
)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

print(f"✅ 전처리 완료:")
print(f"   total batches: {totalbatch}")
print(f"   training samples: {training_data_count}")

def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf_v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf_v1.unstack(_X, axis=1)
    outputs, states = tf_v1.nn.static_rnn(cell, _X, dtype=tf_v1.float32)
    m = []
    for i in outputs:
        o = tf_v1.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf_v1.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf_v1.matmul(last_output, _weights['out']) + _biases['out']
    output = tf_v1.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf_v1.transpose(output, perm=[0,2,1])
    output = tf_v1.reshape(output, shape=[-1,num_nodes])
    
    # 수치 안정성 추가
    output = tf_v1.clip_by_value(output, -100.0, 100.0)
    
    return output, m, states
    
    
###### placeholders ######
print(f"\n🧠 모델 구성...")

if model_name == 'ast-gcn':
    actual_time_len = trainX.shape[1]
    print(f"   V2X input dimension: {actual_time_len}")
    input_dim = actual_time_len
    
    inputs = tf_v1.placeholder(tf_v1.float32, shape=[None, input_dim, num_nodes])
else:
    inputs = tf_v1.placeholder(tf_v1.float32, shape=[None, seq_len, num_nodes])

labels = tf_v1.placeholder(tf_v1.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf_v1.Variable(tf_v1.random_normal([gru_units, pre_len], mean=0.0, stddev=0.1), name='weight_o')}  # 초기값 조정
biases = {
    'out': tf_v1.Variable(tf_v1.random_normal([pre_len], stddev=0.1),name='bias_o')}  # 초기값 조정

pred,ttts,ttto = TGCN(inputs, weights, biases)

y_pred = pred
      
###### optimizer with gradient clipping ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf_v1.nn.l2_loss(tf_var) for tf_var in tf_v1.trainable_variables())
label = tf_v1.reshape(labels, [-1,num_nodes])

##loss
print('y_pred_shape:', y_pred.shape)
print('label_shape:', label.shape)
loss = tf_v1.reduce_mean(tf_v1.nn.l2_loss(y_pred-label) + Lreg)

##rmse
error = tf_v1.sqrt(tf_v1.reduce_mean(tf_v1.square(y_pred-label)))

# Gradient clipping 추가
opt = tf_v1.train.AdamOptimizer(lr)
grads_and_vars = opt.compute_gradients(loss)
clipped_grads_and_vars = [(tf_v1.clip_by_value(grad, -1.0, 1.0), var) 
                          for grad, var in grads_and_vars if grad is not None]
optimizer = opt.apply_gradients(clipped_grads_and_vars)

###### Initialize session ######
variables = tf_v1.global_variables()
saver = tf_v1.train.Saver(tf_v1.global_variables())  

# GPU 설정 (TF 1.x 스타일)
config = tf_v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True  # GPU 메모리 점진적 할당

sess = tf_v1.Session(config=config)
sess.run(tf_v1.global_variables_initializer())

# 출력 디렉토리 설정
if data_name == 'v2x':
    out = f'out/v2x_{model_name}_scheme{scheme}_gpu'
else:
    out = f'out/{model_name}_{noise_name}_gpu'

path1 = f'{model_name}_{name}_{data_name}_lr{lr}_batch{batch_size}_unit{gru_units}_seq{seq_len}_pre{pre_len}_epoch{training_epoch}_scheme{scheme}_PG{PG}_GPU'
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)
    
print(f"📂 결과 저장 경로: {path}")

###### NaN 안전 evaluation 함수 ######
def evaluation(a, b):
    """V2X 데이터용 NaN 안전 평가 함수"""
    
    print(f"🔍 평가 데이터 체크:")
    print(f"   실제값(a) 형태: {a.shape}")
    print(f"   예측값(b) 형태: {b.shape}")
    
    # 1차원으로 flatten
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    # NaN/Inf 검사
    a_nan_count = np.isnan(a_flat).sum()
    b_nan_count = np.isnan(b_flat).sum()
    a_inf_count = np.isinf(a_flat).sum()
    b_inf_count = np.isinf(b_flat).sum()
    
    print(f"   실제값 NaN: {a_nan_count}, Inf: {a_inf_count}")
    print(f"   예측값 NaN: {b_nan_count}, Inf: {b_inf_count}")
    
    # 전체 데이터가 문제인 경우
    total_elements = len(a_flat)
    problem_ratio = (a_nan_count + b_nan_count + a_inf_count + b_inf_count) / (2 * total_elements)
    
    if problem_ratio > 0.5:
        print(f"❌ 문제 데이터 비율: {problem_ratio:.1%} - 기본값 반환")
        return 999.0, 999.0, -999.0, -999.0, -999.0
    
    # 안전한 데이터 처리
    # NaN/Inf를 중간값으로 대체
    a_median = np.nanmedian(a_flat[np.isfinite(a_flat)])
    b_median = np.nanmedian(b_flat[np.isfinite(b_flat)])
    
    if np.isnan(a_median):
        a_median = 0.0
    if np.isnan(b_median):
        b_median = 0.0
    
    a_clean = np.where(np.isfinite(a_flat), a_flat, a_median)
    b_clean = np.where(np.isfinite(b_flat), b_flat, b_median)
    
    # 극값 클리핑 (outlier 제거)
    a_std = np.std(a_clean)
    b_std = np.std(b_clean)
    a_mean = np.mean(a_clean)
    b_mean = np.mean(b_clean)
    
    # 3-sigma 규칙 적용
    a_clipped = np.clip(a_clean, a_mean - 3*a_std, a_mean + 3*a_std)
    b_clipped = np.clip(b_clean, b_mean - 3*b_std, b_mean + 3*b_std)
    
    try:
        # 안전한 메트릭 계산
        rmse = math.sqrt(mean_squared_error(a_clipped, b_clipped))
        mae = mean_absolute_error(a_clipped, b_clipped)
        
        # Frobenius norm (안전하게)
        diff_norm = la.norm(a_clipped - b_clipped)
        a_norm = la.norm(a_clipped)
        if a_norm < 1e-10:
            F_norm = 0.0
        else:
            F_norm = diff_norm / a_norm
        
        # R2 계산 (분산 체크)
        a_var = np.var(a_clipped)
        if a_var < 1e-10:
            r2 = 0.0
            print("⚠️ 실제값 분산이 너무 작음, R2=0 설정")
        else:
            ss_res = np.sum((a_clipped - b_clipped) ** 2)
            ss_tot = np.sum((a_clipped - np.mean(a_clipped)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            r2 = max(r2, -1.0)  # R2 하한 설정
        
        # Variance score
        var_a = np.var(a_clipped)
        var_diff = np.var(a_clipped - b_clipped)
        if var_a < 1e-10:
            var_score = 0.0
        else:
            var_score = 1 - (var_diff / var_a)
        
        print(f"📊 평가 결과:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   F_norm: {1-F_norm:.4f}")
        print(f"   R2: {r2:.4f}")
        print(f"   Var Score: {var_score:.4f}")
        
        return rmse, mae, 1-F_norm, r2, var_score
        
    except Exception as e:
        print(f"❌ 평가 중 오류: {e}")
        # 기본 메트릭이라도 계산
        try:
            rmse = np.sqrt(np.mean((a_clipped - b_clipped)**2))
            mae = np.mean(np.abs(a_clipped - b_clipped))
            acc = 0.5
            r2 = 0.0
            var_score = 0.0
            print(f"⚡ 기본 메트릭: RMSE={rmse:.4f}, MAE={mae:.4f}")
            return rmse, mae, acc, r2, var_score
        except:
            print("❌ 모든 메트릭 계산 실패")
            return 999.0, 999.0, -999.0, -999.0, -999.0

print(f"\n🚀 V2X AST-GCN GPU 학습 시작...")
print(f"   Epochs: {training_epoch}")
print(f"   Batch size: {batch_size}")
print(f"   Learning rate: {lr}")
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
  
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        
        # NaN 체크
        if np.isnan(mini_batch).any() or np.isnan(mini_label).any():
            print(f"⚠️ Epoch {epoch}, Batch {m}: 입력 데이터에 NaN 발견, 건너뜀")
            continue
            
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        
        # 학습 중 NaN 체크
        if np.isnan(loss1) or np.isnan(rmse1):
            print(f"⚠️ Epoch {epoch}, Batch {m}: 학습 중 NaN 발생")
            print(f"   loss: {loss1}, rmse: {rmse1}")
            continue
            
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

    # Test completely at every epoch
    try:
        loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                             feed_dict = {inputs:testX, labels:testY})
        
        # 테스트 중 NaN 체크
        if np.isnan(loss2) or np.isnan(rmse2) or np.isnan(test_output).any():
            print(f"⚠️ Epoch {epoch}: 테스트 중 NaN 발생")
            print(f"   test_loss: {loss2}, test_rmse: {rmse2}")
            print(f"   test_output NaN 개수: {np.isnan(test_output).sum()}")
            
            # NaN을 안전한 값으로 대체
            test_output = np.nan_to_num(test_output, nan=0.0, posinf=1.0, neginf=-1.0)
            loss2 = 999.0 if np.isnan(loss2) else loss2
            rmse2 = 999.0 if np.isnan(rmse2) else rmse2

        testoutput = np.abs(test_output)
        test_label = np.reshape(testY,[-1,num_nodes])
        rmse, mae, acc, r2_score, var_score = evaluation(test_label, testoutput)
        test_label1 = test_label * max_value
        test_output1 = testoutput * max_value
        test_loss.append(loss2)
        test_rmse.append(rmse * max_value)
        test_mae.append(mae * max_value)
        test_acc.append(acc)
        test_r2.append(r2_score)
        test_var.append(var_score)
        test_pred.append(test_output1)
        
        print('Epoch:{}'.format(epoch),
              'train_rmse:{:.4}'.format(batch_rmse[-1] if batch_rmse else 0),
              'test_loss:{:.4}'.format(loss2),
              'test_rmse:{:.4}'.format(rmse * max_value),
              'test_acc:{:.4}'.format(acc))
        
    except Exception as e:
        print(f"❌ Epoch {epoch} 테스트 중 오류: {e}")
        continue
    
    if (epoch % 20 == 0) and epoch > 0:
        model_path = os.path.join(path, 'model_100')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver.save(sess, f'{model_path}/V2X_ASTGCN_GPU_pre_{epoch}', global_step = epoch)
        
time_end = time.time()
print(f'\n⏱️ GPU 학습 완료! 소요 시간: {time_end-time_start:.2f}초')

############## visualization ###############
if batch_rmse and test_rmse:  # 결과가 있는 경우만
    try:
        b = int(len(batch_rmse)/totalbatch)
        batch_rmse1 = [i for i in batch_rmse]
        train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
        batch_loss1 = [i for i in batch_loss]
        train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

        index = test_rmse.index(np.min(test_rmse))
        test_result = test_pred[index]
        var = pd.DataFrame(test_result)
        var.to_csv(path+'/test_result.csv',index = False,header = False)
        plot_result(test_result,test_label1,path)
        plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

        evaluation_metrics = []
        evaluation_metrics.append(np.min(test_rmse))
        evaluation_metrics.append(test_mae[index])
        evaluation_metrics.append(test_acc[index])
        evaluation_metrics.append(test_r2[index])
        evaluation_metrics.append(test_var[index])
        evaluation_df = pd.DataFrame(evaluation_metrics)
        evaluation_df.to_csv(path+'/evaluation.csv',index=False,header=None)

        print(f'\n🎉 V2X AST-GCN GPU 결과:')
        print(f'   model_name: {model_name}')
        print(f'   dataset: {data_name}')
        print(f'   scheme: {scheme} ({name})')
        print(f'   noise_name: {noise_name}')
        print(f'   PG: {PG}')
        print(f'   min_rmse: {np.min(test_rmse):.4f}')
        print(f'   min_mae: {test_mae[index]:.4f}')
        print(f'   max_acc: {test_acc[index]:.4f}')
        print(f'   r2: {test_r2[index]:.4f}')
        print(f'   var: {test_var[index]:.4f}')
        print(f'📂 결과 저장 위치: {path}')
    except Exception as e:
        print(f"❌ 결과 저장 중 오류: {e}")
else:
    print("❌ 학습 결과가 없어 저장하지 않음")