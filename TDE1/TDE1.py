# TDE 1 Algoritmo KNN feito do zero

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import model_selection


# carrega a base
X, y = load_digits(return_X_y=True)
print("X: ", X.shape)
print("y: ", y.shape)


# separa teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nX_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

n_neighbors = 3 # k = número de vizinhos
score = 0 # número de acertos


# Retorna o elemnto que mais se repete na lista 
def returnElementMax(lista):
    return max(set(lista), key=lista.count) # ordena contando os objetos

# distancia euclidiana 
def distanciaEuclidiana(element, x):
    distancia = 01

    for i in range(len(element)):
        distancia += (element[i] - x[i]) ** 2

    return distancia**0.5

# retorna o elemento mais proximo da base atual
def returnElementoProximo(element, x_train, y_train, verificador):
    
    distancia_menor = float('inf') # guarda a menor distancia
    class_xtrain = 0 # guarda a classe do elemento do x_train
    index_elemento_proximo = 0 # guarda o index do elemento mais proximo
 
    for i in range(x_train.shape[0]):
        distancia_Atual = distanciaEuclidiana(element, x_train[i]) # calcula a distancia atual entre o elemento e o elemento do X_train

        if distancia_Atual < distancia_menor and verificador[i]: 
            distancia_menor = distancia_Atual
            class_xtrain = y_train[i] 
            index_elemento_proximo = i
    
    verificador[index_elemento_proximo] = False # retorna como falso para não retornar o mesmo elemento 
    
    # retorna o elemento do x_train que mais se parece com o elemento passado como parametro
    return class_xtrain, verificador



# para cada elemento do X_test vamos comparar com todos os elementos da base de treino
for i in range(X_test.shape[0]):
    
    lista = [0]* n_neighbors # armazena todas as interações dos n_neighbors
    verificador = [True] * len(y_train) # array para verificar e nao pegar sempre o mesmo elemento 

    # percorre a base de treino 3x e cada vez que percorrer vai pegar o elemento mais proximo do X_test atual
    for j in range(n_neighbors):
        lista[j], verificador = returnElementoProximo(X_test[i], X_train, y_train, verificador)
    
    
    # se o elemento atual for igual o elemento do y_test, soma mais um nos acertos
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


