# tgcn.py

# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf  # TF 2.x 호환성
from tensorflow.compat.v1.nn.rnn_cell import RNNCell  # TF 2.x 호환성
import numpy as np
import pandas as pd

# utils.py가 없는 경우를 대비한 calculate_laplacian 함수
def calculate_laplacian(adj_matrix):
    """라플라시안 행렬 계산"""
    adj = np.array(adj_matrix)
    
    # 차수 행렬 계산
    degree = np.diag(np.sum(adj, axis=1))
    
    # 라플라시안 = 차수 - 인접
    laplacian = degree - adj
    
    # 정규화된 라플라시안
    degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(adj, axis=1) + 1e-6))
    normalized_laplacian = np.eye(adj.shape[0]) - degree_inv_sqrt @ adj @ degree_inv_sqrt
    
    # Sparse tensor로 변환
    indices = np.where(normalized_laplacian != 0)
    values = normalized_laplacian[indices]
    shape = normalized_laplacian.shape
    
    sparse_laplacian = tf.SparseTensor(
        indices=np.column_stack(indices),
        values=values.astype(np.float32),
        dense_shape=shape
    )
    
    return sparse_laplacian


class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """
     
    def call(self, inputs, **kwargs):
        pass
     
    def __init__(self, num_units, adj, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):
         
        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = []
        self._adj.append(calculate_laplacian(adj))
      
    @property
    def state_size(self):
        return self._nodes * self._units
     
    @property
    def output_size(self):
        return self._units
     
    def __call__(self, inputs, state, scope=None):
         
        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h
      
    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        ## concat
        x_s = tf.concat([inputs, state], axis=2)
        
        input_size = x_s.get_shape()[2].value
        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])          
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
                 
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            for m in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(m, x0)
            x = tf.reshape(x1, shape=[self._nodes, input_size,-1])
            x = tf.transpose(x,perm=[2,0,1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                'weights', [input_size, output_size], 
                initializer=tf.keras.initializers.glorot_uniform())  # TF 2.x 호환
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                "biases", [output_size], 
                initializer=tf.constant_initializer(bias, dtype=tf.float32))
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x