from re import T
from turtle import shape
import pandas as pd
from matplotlib import widgets
import numpy as np
from sklearn import model_selection, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression  
from warnings import simplefilter
from sklearn import preprocessing
import itertools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tabulate import tabulate
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)

# Carregando a base de dados
database = pd.read_csv("Machine Learning\Formativa3\databaseQuestao1.csv")

print(database)

valores = database.values
colunas = database.columns

X = valores[:, 0:12] # 12 primeiras colunas
y = valores[:, 12] # pega a ultima coluna

print("\nShape de X: ", X.shape)
print("Shape de y: ", y.shape)


# Normalizando os dados
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

# segunda forma de normalização
scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

# X = preprocessing.normalize(X, norm='11', axis=0)
print("\nColuna normalizada\n", X)

# Parametros da função KNN
parameters = [
  {'n_neighbors': [1, 2, 3, 4, 5], 
   'weights': ['uniform', 'distance'], 
   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 
   'p': [1,2]},
  ]

parameters2 = [
  {'max_depth': [3, 5, 10, 20],
   'min_samples_split': [3, 5, 10], 
   'criterion':['entropy', 'gini'], 
   'splitter':['best', 'random']},
  ]

parameters3 = [
    {
     'var_smoothing': [1e-8, 1e-5, 1e-3, 1e-4]
    }
]

def knn(T, parameters):
    model = KNeighborsClassifier()

    result = model_selection.cross_val_score(model, X, y, cv=T)

    gs = GridSearchCV(model, parameters, scoring = 'r2', cv=T, n_jobs=5)

    ## ->
    gs = gs.fit(X, y)
    ## ->

    df = gs.cv_results_
    #print(tabulate(df,headers='keys', tablefmt='psql'))
    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    model=gs.best_estimator_

    result = model_selection.cross_val_score(model, X, y, cv=T)

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

def arvoreDecisao(T, parameters):
    model2 = DecisionTreeClassifier(random_state=42)

    result = model_selection.cross_val_score(model2, X, y, cv=T)

    gs = GridSearchCV(model2, parameters, scoring = 'r2', cv=T, n_jobs=5)

    ## ->
    gs = gs.fit(X, y)
    ## ->

    df = gs.cv_results_
    #print(tabulate(df,headers='keys', tablefmt='psql'))
    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    model2 = gs.best_estimator_

    result = model_selection.cross_val_score(model2, X, y, cv=T)

    # Mostrando a acurácia média e desvio padrão.
    print("\nCross Validation Results %d folds:" % T)
    print("Mean Accuracy: %.5f" % result.mean())
    print("Mean Std: %.5f" % result.std())

    # Calculando a predição para exemplo de teste
    y_pred = model_selection.cross_val_predict(model2, X, y, cv=T)

    # Calculando para cada instância de teste a probabilidade de cada classe
    predicted_proba=model_selection.cross_val_predict(model2, X, y, cv=T, method='predict_proba')

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

def naivebayes(T, parameters):
    model = GaussianNB()

    result = model_selection.cross_val_score(model, X, y, cv=T)

    gs = GridSearchCV(model, parameters, scoring = 'r2', cv=T, n_jobs=5)

    ## ->
    gs = gs.fit(X, y)
    ## ->

    df = gs.cv_results_
    #print(tabulate(df,headers='keys', tablefmt='psql'))
    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    model=gs.best_estimator_

    result = model_selection.cross_val_score(model, X, y, cv=T)

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
T = 10
print("\n==== Função KNN ====")

knn(T, parameters)

print("\n==== Função Arvore de Decisão ====")

arvoreDecisao(T, parameters2)

print("\n==== Naive Bayes ====")
naivebayes(T, parameters3)