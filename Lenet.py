# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

# bibliotecas
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
from datetime import datetime

# calcula o tempo
todo_algoritmo_ini = time.time()

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]

modelo = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'),
    keras.layers.AveragePooling2D(),
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
    keras.layers.AveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='tanh'),
    keras.layers.Dense(84, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')])

modelo.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['acc'])

# contagem do tempo
tempo_ini = time.time()

modelo.fit(train_x, train_y, epochs=100, validation_data=(val_x, val_y))

# finalizacao do tempo
tempo_fim = time.time()

# tempo de teste
teste_ini = time.time()

modelo.evaluate(test_x, test_y)

teste_fim = time.time()


todo_algoritmo_fim = time.time()

print('Tempo de Treinamento: ' + str(tempo_fim - tempo_ini))
print('Tempo do Teste: ' + str(teste_fim - teste_ini))
print('Tempo Total: '+ str(todo_algoritmo_fim - todo_algoritmo_ini))