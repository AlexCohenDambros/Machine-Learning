# probability = true -> usado quando quero uma probabilidade na saida de uma classe 
# random_state 42 
from sklearn.svm import SVC
import urllib
import urllib.request as request
from turtle import shape
import pandas as pd
from matplotlib import widgets
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_digits

np.random.seed(42)

# carrega a base
X, y = load_digits(return_X_y=True)
print("X: ", X.shape)
print("y: ", y.shape)


# separa teste e treino
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

print("\nX_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_val.shape)
print("y_test: ", y_val.shape)

# Parametros da função KNN
parameters1 = [
   {'C': [1,5,10,100,200], 'kernel': ['linear'], 
    'C': [1,5,10,50,100,200], 'kernel': ['poly'],
    'C': [1,5,10,100,150,500], 'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale'], 'kernel':['rbf']
    },
   ]


def svc(T, parameters_here):
    clf = SVC(probability=True)

    gs = GridSearchCV(clf, parameters_here, scoring = 'accuracy', cv=T, n_jobs=5)

    ## ->
    gs.fit(X_val, y_val)
    ## ->

    df = gs.cv_results_
    #print(tabulate(df,headers='keys', tablefmt='psql'))
    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    model=gs.best_estimator_

    result = model_selection.cross_val_score(model, X_train, y_train, cv=T)

    # Mostrando a acurácia média e desvio padrão.
    print("\nCross Validation Results %d folds:" % T)
    print("Mean Accuracy: %.5f" % result.mean())
    print("Mean Std: %.5f" % result.std())

    # Calculando a predição para exemplo de teste
    y_pred = model_selection.cross_val_predict(model, X, y, cv=T)

    # Calculando para cada instância de teste a probabilidade de cada classe
    predicted_proba=model_selection.cross_val_predict(model, X, y, cv=T, method='predict_proba')

    # Calculando a precisão na base de teste
    precision=precision_score(y, y_pred, average='weighted')
    print("Precision = %.3f " % precision)

    # Calculando a revocação na base de teste
    recall=recall_score(y, y_pred, average='weighted')
    print("Recall = %.3f " % recall)

    # Calculando f1 na base de teste
    f1=f1_score(y, y_pred, average='weighted')
    print("F1 = %.3f " % f1)

    # Exemplo mostrando o resultado previsto para a primeira instância de teste
    print("Primeira instância na base de teste foi considerada como da classe: %d" % y_pred[0])

    # Exemplo abaixo mostrando para a primeira instância de teste a probabilidade de cada classe
    print("Probabilidade de cada classe para a primeira instância: ", predicted_proba[0])

    # Calculando a matriz de confusão
    print("Matriz de Confusão:")
    matrix = confusion_matrix(y, y_pred)
    print(matrix)


# Treinando o classificador
T = 10 # Número de folds

print("\n==== Função SVC ====")

svc(T, parameters1)
