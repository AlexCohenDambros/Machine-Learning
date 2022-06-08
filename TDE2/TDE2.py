# Feature Extraction (Handcrafted and Deep Features)

import cv2
import re
import math
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


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)


# Load Inception_v3 pretrained on ImageNet dataset
model = InceptionV3(include_top=False, weights='imagenet',
                    pooling='avg', input_tensor=Input(shape=(299, 299, 3)))

# List of paths
file_list = []
file_list.append(os.listdir(r"Base/humanos"))
file_list.append(os.listdir(r"Base/praia"))
file_list.append(os.listdir(r"Base/obras"))
file_list.append(os.listdir(r"Base/onibus"))
file_list.append(os.listdir(r"Base/dino"))
file_list.append(os.listdir(r"Base/elefante"))
file_list.append(os.listdir(r"Base/flores"))
file_list.append(os.listdir(r"Base/cavalos"))
file_list.append(os.listdir(r"Base/montanhas"))
file_list.append(os.listdir(r"Base/comida"))

# general path
path = 'Base/'

# list of classes
class_names = ['humanos', 'praia', 'obras', 'onibus', 'dino',
               'elefante', 'flores', 'cavalos', 'montanhas', 'comida']


file_exists_X_deep = os.path.exists('X_deep.csv')
file_exists_y = os.path.exists('y.csv')

if file_exists_X_deep and file_exists_y:
    X_deep = pd.read_csv('X_deep.csv')
    y = pd.read_csv('y.csv')

    X_deep = X_deep.fillna(0)
    y = y.fillna(0)

else:
    X_deep = []
    y = []

# ======== Parametros das funcoes de Machine Learning ========

parametrosKNN = [
    {'n_neighbors': list(range(1, 10, 1)),
     'weights': ['uniform', 'distance'],
     'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
     'p': [1, 2],
     'leaf_size': list(range(2, 20, 2))},
]

parametrosDecisionTrees = [
    {'max_depth': list(range(3, 60, 3)),
     'min_samples_split': list(range(5, 25, 5)),
     'criterion': ['entropy', 'gini'],
     'splitter':['best', 'random']},
]

parametrosNaiveBayes = [
    {
        'var_smoothing': [1e-9, 1e-8, 1e-5, 1e-4, 1e-3]
    }
]

parametersSVM = [
    {'C': [1, 5, 10, 100, 200], 'kernel': ['linear'],
     'C': [1, 5, 10, 50, 100, 200], 'kernel': ['poly'],
     'C': [1, 5, 10, 100, 150, 500], 'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'], 'kernel':['rbf']
     },
]

parametersRandomForest = [
    {'n_estimators': list(range(10, 190, 20)),
     'max_depth': list(range(3, 30, 3)),
     'min_samples_split': list(range(5, 25, 5)),
     'criterion': ['gini', 'entropy']
    }
]


parametersBaggind = [
    {'n_estimators': list(range(10, 100, 10)),
     'max_samples': list(range(2, 20, 2)),
     'bootstrap': [True, False],
     'bootstrap_features': [True, False],
     'base_estimator': [None, DecisionTreeClassifier(criterion='entropy', max_depth=5), DecisionTreeClassifier(criterion='entropy', max_depth=7), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=3, weights='distance')]
     }
]


# # Feature extraction
# for classes_files, classe in zip(file_list, range(10)):
#     for i in range(100):
#         name = str(path) + str(class_names[classe]
#                                ) + str('/') + str(classes_files[i])
#         print(name)
#         y.append(classe)

# # Extract deep features using InceptionV3 pretrained model
#         imagem = cv2.imread(name)
#         img = cv2.resize(imagem, (299, 299))
#         xd = image.img_to_array(img)
#         xd = np.expand_dims(xd, axis=0)
#         xd = preprocess_input(xd)
#         deep_features = model.predict(xd)
#         print(deep_features.shape)

#         X_image_aux = []
#         for aux in deep_features:
#             X_image_aux = np.append(X_image_aux, np.ravel(aux))

#         deep_features = [i for i in X_image_aux]

#         X_deep.append(deep_features)

# # Saving the extracted features (deep) in a csv file
# df = pd.DataFrame(X_deep)
# df.to_csv('X_deep.csv', header=False, index=False)

# # Saving the classes in a csv file
# df_class = pd.DataFrame(y)
# df_class.to_csv('y.csv', header=False, index=False)


# ===================================================================================


# Labels
y = pd.read_csv('y.csv', header=None)
y = y.to_numpy()
y = np.ravel(y)
print(y.shape)

# deep features
X = pd.read_csv('X_deep.csv', header=None)
X = X.to_numpy()
print(X.shape)


# ===================================================================================

# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (60%) e teste (40%), estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.4, random_state=42, stratify=y)


def plotResultados(score, matrix):

    # apresenta os resultados
    print("Accuracy = %.2f " % score)
    print("Confusion Matrix:")
    print(matrix)


def kNN(parameters):

    # Treina o classificador
    clfa = KNeighborsClassifier()

    clfa = GridSearchCV(clfa, parameters, scoring='r2', n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_
    print(best_parameters)

    # testa usando a base de testes
    predicted = clfa.predict(X_test)
    predp = clfa.predict_proba(X_test)

    # calcula a acurÃ¡cia na base de teste
    score = clfa.score(X_test, y_test)

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return predicted, predp, score, matrix, best_parameters


def decisionTrees(parameters):

    # Treina o classificador
    clfa = DecisionTreeClassifier(random_state=42)

    clfa = GridSearchCV(clfa, parameters, scoring='r2', n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)
    predp = clfa.predict_proba(X_test)

    # calcula a acurÃ¡cia na base de teste
    score = clfa.score(X_test, y_test)

    # calcula a matriz de confusÃ£o
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return predicted, predp, score, matrix, best_parameters


def sVM(parameters):

    # Treina o classificador
    clfa = SVC(probability=True)

    clfa = GridSearchCV(clfa, parameters, scoring='accuracy', n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)
    predp = clfa.predict_proba(X_test)

    # calcula a acuracia na base de teste
    score = clfa.score(X_test, y_test)

    # calcula a matriz de confusao
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return predicted, predp, score, matrix, best_parameters


def naiveBayes(parameters):

    # Treina o classificador
    clfa = GaussianNB()

    clfa = GridSearchCV(clfa, parameters, scoring='r2', n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)
    predp = clfa.predict_proba(X_test)

    # calcula a acurÃ¡cia na base de teste
    score = clfa.score(X_test, y_test)

    # calcula a matriz de confusÃ£o
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return predicted, predp, score, matrix, best_parameters


def randomForest(parameters):

    # Treina o classificador
    clfa = RandomForestClassifier(random_state=42)

    clfa = GridSearchCV(clfa, parameters, scoring='r2', n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)
    predp = clfa.predict_proba(X_test)

    # calcula a acurÃ¡cia na base de teste
    score = clfa.score(X_test, y_test)

    # calcula a matriz de confusÃ£o
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return predicted, predp, score, matrix, best_parameters


def bagging(parameters):
    # Treina o classificador
    clfa = BaggingClassifier(random_state=42)

    clfa = GridSearchCV(clfa, parameters, scoring='r2', n_jobs=5)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    # testa usando a base de testes
    predicted = clfa.predict(X_test)
    predp = clfa.predict_proba(X_test)

    # calcula a acurÃ¡cia na base de teste
    score = clfa.score(X_test, y_test)

    # calcula a matriz de confusÃ£o
    matrix = confusion_matrix(y_test, predicted)

    print("\nConcluido!")

    return predicted, predp, score, matrix, best_parameters

melhoresResultados = {}

print("\n==== Executando --> KNN ====")
predicted, predp, score, matrix, best_parameters = kNN(parametrosKNN)
melhoresResultados["KNN"] = [predicted, predp, score, matrix, best_parameters]

print("\n==== Executando --> Arvore de Decisao ====")
predicted, predp, score, matrix, best_parameters = decisionTrees(parametrosDecisionTrees)
melhoresResultados["DecisionTrees"] = [predicted, predp, score, matrix, best_parameters]

print("\n==== Executando --> SVM ====")
predicted, predp, score, matrix, best_parameters = sVM(parametersSVM)
melhoresResultados["SVM"] = [predicted, predp, score, matrix, best_parameters]

print("\n==== Executando --> Naive Bayes ====")
predicted, predp, score, matrix, best_parameters = naiveBayes(parametrosNaiveBayes)
melhoresResultados["NaiveBayes"] = [predicted, predp, score, matrix, best_parameters]

print("\n==== Executando --> Random Forest ====")
predicted, predp, score, matrix, best_parameters = randomForest(
    parametersRandomForest)
melhoresResultados["RandomForest"] = [
    predicted, predp, score, matrix, best_parameters]

print("\n==== Executando --> Bagging ====")
predicted, predp, score, matrix, best_parameters = bagging(parametersBaggind)
melhoresResultados["Bagging"] = [
    predicted, predp, score, matrix, best_parameters]

for key in melhoresResultados:
    print("Classification accuracy {}: {}".format(
        key, melhoresResultados[key][2]))

# ===================================================================================

# Plot mistakes (images)

# print(predicted.shape)
# for i in range(len(predicted)):
#     if (predicted[i] != y_test[i]):
#         dist = 1
#         j = 0
#         while (j < len(X) and dist != 0):
#            dist = np.linalg.norm(X[j]-X_test[i])
#            j += 1
#        print("Label:", y[j-1], class_names[y[j-1]], "  /  Prediction: ",
#              predicted[i], class_names[predicted[i]], predp[i][predicted[i]])
#         name = path + \
#            str(class_names[y[j-1]]) + "/" + str(j) + ".jpg"
#         print(name)
#         im = cv2.imread(name)
#         cv2.imshow("TDE2", im)
#        print("=============================================================================")
