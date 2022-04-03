# Programar o KNN na mão

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import model_selection

X, y = load_digits(return_X_y=True)
print("Formato de X: ", X.shape)
print("Formato de y: ", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nFormato de X_train: ", X_train.shape)
print("Formato de y_train: ", y_train.shape)
print("Formato de X_test: ", X_test.shape)
print("Formato de y_test: ", y_test.shape)

n_neighbors = 3 # k
contar = 0

def returnElementMax(lista):
    return max(set(lista), key=lista.count)

def distancia(element, x):
    dist = 0

    for i in range(len(element)):
        dist += (element[i] - x[i]) ** 2

    return dist**0.5

def returnElementoProximo(element, x_train, y_train, verificador):
    
    min_distance = float('inf')
    class_most_close = 0
    index_elemento_proximo = 0 

    for i in range(x_train.shape[0]):
        current_distance = distancia(element, x_train[i])

        if current_distance < min_distance and verificador[i]:
            min_distance = current_distance
            class_most_close = y_train[i]
            index_elemento_proximo = i
    
    verificador[index_elemento_proximo] = False
    return class_most_close, verificador


for i in range(X_test.shape[0]):

    lista = [0]* n_neighbors
    verificador = [True] * len(y_train)

    for j in range(n_neighbors):
        lista[j], verificador = returnElementoProximo(X_test[i], X_train, y_train, verificador)
    
    
    if returnElementMax(lista) == y_test[i]:
        contar+=1

print('Acertos: ', contar, "Rate: ", contar/np.shape(y_test)[0])