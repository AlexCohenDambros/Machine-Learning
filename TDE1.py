# TDE 1 Algoritmo KNN feito do zero

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import model_selection

X, y = load_digits(return_X_y=True)
print("X: ", X.shape)
print("y: ", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nX_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

n_neighbors = 3 # k
score = 0

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
        score+=1
    
print("\n ====== Algoritmo KNN feito do zero! ====== \n")
print("Accuracy: ", score/np.shape(y_test)[0], "\nAcertos total: ", score)
      
print("\n ====== Algoritmo KNN usando sklearn ====== \n")

# Declaram o modelo (classificador)
clf = KNeighborsClassifier(n_neighbors=3)

# Treinamento do modelo 
clf = clf.fit(X_train, y_train)

# Avaliar treinamento (calcula a taxa de acerto)
score2 = clf.score(X_test, y_test)

# Predição do modelo
predicted = clf.predict(X_test)

# Calcula a matriz de confusão
matrix = confusion_matrix(y_test, predicted)

print("Accuracy = %.5f" % score2)
print("Confusion Matrix: \n", matrix) # na diagonal eh os acertos


