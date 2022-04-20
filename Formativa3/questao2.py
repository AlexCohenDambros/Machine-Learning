from email.mime import base
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import model_selection, preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_diabetes
from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed(42)

# Carregando a base de dados
database = pd.read_csv("Machine Learning\Formativa3\databaseQuestao2.csv")

# Conversao categoricos para numericos
database['sex'] = database['sex'].map({'female': 0, 'male': 1})
database['smoker'] = database['smoker'].map({'no': 0, 'yes': 1})
database['region'] = database['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

print(database)

valores = database.values
colunas = database.columns

X = valores[:, 0:6] # 6 primeiras colunas
y = valores[:, 6] # pega a ultima coluna

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

# separando uma parte para base de validação (20%)
X, X_val, y, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

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
   'criterion':['absolute_error', 'squared_error'], 
   'splitter':['best', 'random']},
  ]


def knn(T, parameters):
    model = KNeighborsRegressor()

    result = model_selection.cross_val_score(model, X, y, cv=T)

    gs = GridSearchCV(model, parameters, scoring = 'r2', cv=T, n_jobs=5)

    ## ->
    gs = gs.fit(X_val, y_val)
    ## ->

    df = gs.cv_results_
    #print(tabulate(df,headers='keys', tablefmt='psql'))
    print(gs.best_params_)

    # Definindo a técnica a ser utilizada
    model=gs.best_estimator_

    result = model_selection.cross_val_score(model, X, y, cv=T)

   # Mostrando R2 médio e desvio padrão calculados na validação cruzada.
    print("\nCross Validation Results %d folds:" % T)
    print("R2 médio: %.5f" % result.mean())
    print("Mean Std: %.5f" % result.std())

    # Calculando o valor para cada exemplo de teste
    y_pred = model_selection.cross_val_predict(model, X, y, cv=T)

    # Exemplo mostrando o resultado previsto para a primeira instância de teste
    print("Primeira instância na base de teste apresenta valor diabetes: %d" % y_pred[0])

    # Calculando o erro médio absoluto
    mae=mean_absolute_error(y, y_pred)
    print("Mean Absolute Error (MAE) calculado na base de teste: %.5f" % mae)


def arvoreDecisao(T, parameters):
    model2 = DecisionTreeRegressor(random_state=42)

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

    # Mostrando R2 médio e desvio padrão calculados na validação cruzada.
    print("\nCross Validation Results %d folds:" % T)
    print("R2 médio: %.5f" % result.mean())
    print("Mean Std: %.5f" % result.std())

    # Calculando o valor para cada exemplo de teste
    y_pred = model_selection.cross_val_predict(model2, X, y, cv=T)

    # Exemplo mostrando o resultado previsto para a primeira instância de teste
    print("Primeira instância na base de teste apresenta valor diabetes: %d" % y_pred[0])

    # Calculando o erro médio absoluto
    mae=mean_absolute_error(y, y_pred)
    print("Mean Absolute Error (MAE) calculado na base de teste: %.5f" % mae)

# Treinando o classificador

T = 10
print("\n==== Função KNN ====")

knn(T, parameters)

print("\n==== Arvore de Regressão ====")
arvoreDecisao(T, parameters2)
