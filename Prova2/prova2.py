# Carregando a base
from sklearn.datasets import load_breast_cancer

import cv2
from matplotlib.pyplot import cla
from skimage import feature
from skimage.feature import hog
import os
import os.path

# Example: how to load the csv files (features and labels)
import pandas as pd
import numpy as np

# Treinamento de classificadores
import urllib
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)


X, y = load_breast_cancer(return_X_y=True)

# separa teste e treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y)

# ======== Parametros das funcoes de Machine Learning ========

parametrosNaiveBayes = [
    {
        'var_smoothing': [1e-9, 1e-8, 1e-5, 1e-4, 1e-3]
    }
]

parametersSVC = [
    {'C': [1, 5, 10, 100, 200], 'kernel': ['linear'],
     'C': [1, 5, 10, 50, 100, 200], 'kernel': ['poly'],
     'C': [1, 5, 10, 100, 150, 500], 'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'], 'kernel':['rbf']
     },
]

parametersRandomForest = [
    {'n_estimators': range(80, 200, 20),
     'max_depth': range(3, 30, 3),
     'min_samples_split': range(5, 25, 5),
     'criterion': ['gini', 'entropy']
     }
]


parametersBaggind = [
    {'n_estimators': list(range(80, 200, 40)),
     'base_estimator': [None, DecisionTreeClassifier(criterion='entropy', max_depth=5), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=3, weights='distance')]
     }
]

parametersMLP = [
    {
        'alpha': [0.0001, 0.001, 0.005],
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200, 300, 400],
        'early_stopping': [True, False]
    }
]


def plotResultados(matrix):
    print("Confusion Matrix:")
    print(matrix)


def svc(parameters, folds):

   # Treina o classificador
    clfa = SVC(probability=True)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds,  n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return score, matrix, best_parameters


def randomForestClassifier(parameters, folds):

    # Treina o classificador
    clfa = RandomForestClassifier(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return score, matrix, best_parameters


def baggingClassifier(parameters, folds):

    # Treina o classificador
    clfa = BaggingClassifier(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return score, matrix, best_parameters


def mlpClassifier(parameters, folds):

    # Treina o classificador
    clfa = MLPClassifier(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    result = model_selection.cross_val_score(clfa, X_train, y_train, cv=folds)

    # calcula a acuracia na base de teste
    score = result.mean()

    # Calculando a predição para exemplo de teste
    y_pred = model_selection.cross_val_predict(clfa, X, y, cv=folds)

    # calcula a matriz de confusao
    matrix = confusion_matrix(y, y_pred)

    print("\nConcluido!")

    return score, matrix, best_parameters


def processing_algorithms():
    folds = 5

    print("\n==== Executando --> SVC ====")
    score, matrix, best_parameters = svc(parametersSVC, folds)
    melhoresResultados["SVM"] = [score, matrix, best_parameters]

    print("\n==== Executando --> Random Forest ====")
    score, matrix, best_parameters = randomForestClassifier(
        parametersRandomForest, folds)
    melhoresResultados["RandomForest"] = [score, matrix, best_parameters]

    print("\n==== Executando --> Bagging ====")
    score, matrix, best_parameters = baggingClassifier(parametersBaggind, folds)
    melhoresResultados["Bagging"] = [score, matrix, best_parameters]

    print("\n==== Executando --> MLP_Classifier ====")
    score, matrix, best_parameters = mlpClassifier(parametersMLP, folds)
    melhoresResultados["MLP_Classifier"] = [score, matrix, best_parameters]

    print("================================")
    print("\nResultados encontrados: \n")
    melhorAcuracia = ["", 0]
    for key in melhoresResultados:
        print("Classification accuracy {}: {}".format(
            key, melhoresResultados[key][0]))
        if melhorAcuracia[1] < melhoresResultados[key][0]:
            melhorAcuracia[0] = key
            melhorAcuracia[1] = melhoresResultados[key][0]

    print("================================")
    print("\nMelhor resultado foi do: {} \nAcuracia de: {}".format(
        melhorAcuracia[0], melhorAcuracia[1]))
    for key in melhoresResultados:
        if key == melhorAcuracia[0]:
            score, matrix, best_parameters = melhoresResultados[key][
                0], melhoresResultados[key][1], melhoresResultados[key][2]

    print("\nParametros utilizados: {}".format(best_parameters))
    plotResultados(matrix)

# ===================================================================================


melhoresResultados = {}
print("\n ========== Questao 1 da prova ==========")
processing_algorithms()
