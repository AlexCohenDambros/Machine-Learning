from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing

import warnings
from warnings import simplefilter
warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)

random_state = 42
np.random.seed(42)


X, y = load_breast_cancer(return_X_y=True)

scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

# separa teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

# ======== Parametros das funcoes de Machine Learning ========

parametersSVC = [
    {'kernel': ["linear", "poly", "rbf", "sigmoid"],
     'gamma': ['scale', 'auto']
     }
]

parametersRandomForest = [
    {'n_estimators': range(80, 200, 20),
     'max_depth': range(3, 30, 3),
     'min_samples_split': range(5, 25, 5),
     'criterion': ['gini', 'entropy']
     }
]

parametersBaggind = [
    {'n_estimators': range(80, 200, 40),
     'base_estimator': [None, DecisionTreeClassifier(criterion='entropy', max_depth=5), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=3, weights='distance')]
     }
]

parametersMLP = [
    {'solver': ['adam', 'sgd'],
     'hidden_layer_sizes': [4, 7, (6, 8)],
     'max_iter': [100, 300, 500, 800, 1300],
     }
]

# ================= Funções de aprendizagem - Classificadores =================

def plotResultados(matrix):
    print("Confusion Matrix:")
    print(matrix)


def svc(parameters, folds):

   # Treina o classificador
    clfa = SVC(probability=True, random_state=42)

    clfa = GridSearchCV(clfa, parameters, scoring='accuracy', cv=folds,  n_jobs=-1)

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
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=-1)

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
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=-1)

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
        clfa, parameters, scoring='accuracy', cv=folds, n_jobs=-1)

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

# =========================================================================

def processing_algorithms():
    folds = 5

    print("\n==== Executando --> SVC ====")
    score, matrix, best_parameters = svc(parametersSVC, folds)
    melhoresResultados["SVM"] = [score, matrix, best_parameters]

    print("\n==== Executando --> Random Forest ====")
    score, matrix, best_parameters = randomForestClassifier(parametersRandomForest, folds)
    melhoresResultados["RandomForest"] = [score, matrix, best_parameters]

    print("\n==== Executando --> Bagging ====")
    score, matrix, best_parameters = baggingClassifier(
        parametersBaggind, folds)
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