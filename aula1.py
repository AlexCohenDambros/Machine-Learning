import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits 
import pandas as pd 

# Carregando a base de dados 
X, y = load_digits(return_X_y = True) # X (Atributos), y ()

print(X.shape)
print(y.shape)

# 1797 instancias, para cada vetor tem 64 entradas.

# Treinamento de avaliação (Holdout)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42, stratify=y)

# Declaram o modelo (classificador)
clf = KNeighborsClassifier(n_neighbors=3)

# Treinamento do modelo 
clf = clf.fit(X_train, y_train)

# Avaliar treinamento (calcula a taxa de acerto)
score = clf.score(X_test, y_test)

# Predição do modelo
predicted = clf.predict(X_test)

# Calcula a matriz de confusão
matrix = confusion_matrix(y_test, predicted)

print("Accuracy = %.5f" % score)
print("Confusion Matrix: \n", matrix) # na diagonal eh os acertos