# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

import numpy as np
import time # para calcular o tempo de execução

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score

from dbn import SupervisedDBNClassification
# use "from dbn import SupervisedDBNClassification" for computations on CPU with numpy

todo_algoritmo_ini = time.time()

# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# contagem do tempo
tempo_ini = time.time()

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=100,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# finalizacao do tempo
tempo_fim = time.time()

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# tempo de teste
teste_ini = time.time()

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

teste_fim = time.time()

todo_algoritmo_fim = time.time()

print('Tempo de Treinamento: ' + str(tempo_fim - tempo_ini))
print('Tempo do Teste: ' + str(teste_fim - teste_ini))
print('Tempo Total: '+ str(todo_algoritmo_fim - todo_algoritmo_ini))