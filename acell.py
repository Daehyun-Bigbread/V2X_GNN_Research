#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:10:22 2019

@author: dhh

Modified for V2X data support - 2025-06-03
V2X 데이터 전용, TensorFlow 2.x 호환
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf  # TF 2.x 호환성
dim = 20


def load_assist_data(dataset):
    """기존 Shenzhen 데이터 로딩 (호환성을 위해 유지)"""
    sz_adj = pd.read_csv('%s_adj.csv'%dataset, header=None)
    adj = np.mat(sz_adj)
    data = pd.read_csv('sz_speed.csv')
    return data, adj

def load_v2x_data(dataset='v2x'):
    """
    V2X 데이터 로딩
    
    Args:
        dataset (str): 데이터셋 이름 (기본값: 'v2x')
    
    Returns:
        data (DataFrame): 속도 데이터 (시간 × 차량)
        adj (matrix): 인접행렬 (차량 × 차량)
        poi_data (ndarray): POI 데이터 (차량 × 특성)
        weather_data (ndarray): Weather 데이터 (시간 × 특성)
    """
    print(f"🚗 V2X 데이터 로딩: {dataset}")
    
    # 1. 인접행렬 로딩
    adj_df = pd.read_csv(f'v2x_astgcn_data/{dataset}_adj.csv', header=None)
    adj = np.mat(adj_df)
    print(f"   ✅ 인접행렬: {adj.shape}")
    
    # 2. 속도 데이터 로딩
    data = pd.read_csv(f'v2x_astgcn_data/{dataset}_speed.csv', header=None)
    print(f"   ✅ 속도 데이터: {data.shape}")
    
    # 3. POI 데이터 로딩
    poi_df = pd.read_csv(f'v2x_astgcn_data/{dataset}_poi.csv', header=None)
    poi_data = poi_df.values
    print(f"   ✅ POI 데이터: {poi_data.shape}")
    
    # 4. Weather 데이터 로딩
    weather_df = pd.read_csv(f'v2x_astgcn_data/{dataset}_weather.csv', header=None)
    weather_data = weather_df.values
    print(f"   ✅ Weather 데이터: {weather_data.shape}")
    
    return data, adj, poi_data, weather_data

def preprocess_data(data1, time_len, train_rate, seq_len, pre_len, model_name, scheme, poi_data=None, weather_data=None):
    """
    V2X 데이터 전처리 (차원 수정 버전)
    """
    print(f"🔄 V2X 데이터 전처리 시작:")
    print(f"   📊 데이터 형태: {data1.shape}")
    print(f"   🎯 모델: {model_name}")
    print(f"   🔧 Scheme: {scheme}")
    
    train_size = int(time_len * train_rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]
    
    print(f"   ✂️ 훈련 데이터: {train_data.shape}")
    print(f"   ✂️ 테스트 데이터: {test_data.shape}")
    
    if model_name == 'tgcn':
        ################TGCN###########################
        print("   🤖 TGCN 전처리...")
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]
            trainX.append(a1[0 : seq_len])
            trainY.append(a1[seq_len : seq_len + pre_len])
        for i in range(len(test_data) - seq_len -pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            testX.append(b1[0 : seq_len])
            testY.append(b1[seq_len : seq_len + pre_len])
            
    else:
        ################AST-GCN (V2X 데이터 사용)###########################
        print("   🤖 AST-GCN 전처리 (V2X 데이터)...")
        
        # V2X POI/Weather 데이터 검증
        if poi_data is None or weather_data is None:
            raise ValueError("❌ V2X 데이터를 위해서는 poi_data와 weather_data가 필수입니다!")
        
        # V2X POI 데이터 전처리
        sz_poi = np.transpose(poi_data)  # (특성 × 차량)
        sz_poi_max = np.max(sz_poi) if sz_poi.size > 0 else 1.0
        sz_poi_nor = sz_poi / sz_poi_max if sz_poi_max > 0 else sz_poi
        print(f"   📊 V2X POI 정규화: {sz_poi_nor.shape}, max={sz_poi_max:.3f}")
        
        # V2X Weather 데이터 전처리
        sz_weather = np.array(weather_data)  # (시간 × 특성)
        sz_weather_max = np.max(sz_weather) if sz_weather.size > 0 else 1.0
        sz_weather_nor = sz_weather / sz_weather_max if sz_weather_max > 0 else sz_weather
        print(f"   📊 V2X Weather 정규화: {sz_weather_nor.shape}, max={sz_weather_max:.3f}")
        
        num_vehicles = data1.shape[1]
        weather_features = sz_weather_nor.shape[1]
        poi_features = sz_poi_nor.shape[0]
        
        # 훈련/테스트용 Weather 데이터 분할
        sz_weather_nor_train = sz_weather_nor[0:train_size]
        sz_weather_nor_test = sz_weather_nor[train_size:time_len]
        
        print(f"   📊 차원 정보:")
        print(f"     seq_len: {seq_len}")
        print(f"     weather_features: {weather_features}")
        print(f"     poi_features: {poi_features}")
        print(f"     vehicles: {num_vehicles}")
        
        # Scheme에 따른 데이터 구성
        if scheme == 1:  # POI만 추가
            print("   🏢 Scheme 1: V2X POI만 사용")
            expected_dim = seq_len + poi_features
            print(f"   📐 예상 차원: {seq_len} + {poi_features} = {expected_dim}")
            
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len], sz_poi_nor))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len - pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len], sz_poi_nor))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
                
        elif scheme == 2:  # Weather만 추가
            print("   🌤️ Scheme 2: V2X Weather만 사용")
            expected_dim = seq_len + weather_features
            print(f"   📐 예상 차원: {seq_len} + {weather_features} = {expected_dim}")
            
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                # Weather 데이터: 시퀀스 시작 시점의 Weather 특성
                weather_point = sz_weather_nor_train[i:i+1]  # (1, 14) - 시작 시점만
                # 모든 차량에 동일한 Weather 적용
                weather_broadcast = np.repeat(weather_point, num_vehicles, axis=1).reshape(weather_features, num_vehicles)
                
                a = np.row_stack((a1[0:seq_len], weather_broadcast))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
                
            for i in range(len(test_data) - seq_len - pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                # Weather 데이터: 시퀀스 시작 시점의 Weather 특성
                weather_point = sz_weather_nor_test[i:i+1]  # (1, 14)
                # 모든 차량에 동일한 Weather 적용
                weather_broadcast = np.repeat(weather_point, num_vehicles, axis=1).reshape(weather_features, num_vehicles)
                
                b = np.row_stack((b1[0:seq_len], weather_broadcast))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
                
        else:  # scheme == 3: POI + Weather
            print("   🏢🌤️ Scheme 3: V2X POI + Weather 모두 사용")
            expected_dim = seq_len + weather_features + poi_features
            print(f"   📐 예상 차원: {seq_len} + {weather_features} + {poi_features} = {expected_dim}")
            
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                # Weather 데이터: 시퀀스 시작 시점의 Weather 특성
                weather_point = sz_weather_nor_train[i:i+1]  # (1, 14)
                # 모든 차량에 동일한 Weather 적용
                weather_broadcast = np.repeat(weather_point, num_vehicles, axis=1).reshape(weather_features, num_vehicles)
                
                a = np.row_stack((a1[0:seq_len], weather_broadcast, sz_poi_nor))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
                
            for i in range(len(test_data) - seq_len - pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                # Weather 데이터: 시퀀스 시작 시점의 Weather 특성
                weather_point = sz_weather_nor_test[i:i+1]  # (1, 14)
                # 모든 차량에 동일한 Weather 적용
                weather_broadcast = np.repeat(weather_point, num_vehicles, axis=1).reshape(weather_features, num_vehicles)
                
                b = np.row_stack((b1[0:seq_len], weather_broadcast, sz_poi_nor))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])

    # 결과 변환
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    
    print(f"   ✅ V2X 전처리 완료:")
    print(f"     trainX: {trainX1.shape}")
    print(f"     trainY: {trainY1.shape}")
    print(f"     testX: {testX1.shape}")
    print(f"     testY: {testY1.shape}")
    
    return trainX1, trainY1, testX1, testY1
    
# 기존 Unit 클래스들은 그대로 유지 (TF 2.x 호환성 추가)
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