# Ler direto da URL
import urllib
import urllib.request as request
from turtle import shape
import pandas as pd
from matplotlib import widgets
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from warnings import simplefilter
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(42)

# Carregando a base de dados
# Carregando a base de dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
raw_data = urllib.request.urlopen(url)

# Carrega arquivo como uma matriz
dataset = np.loadtxt(raw_data, delimiter=",")

# Separa atributos de entrada em X e as classes em y
# Já ignora o ID da instância
X = dataset[:,1:10]
y = dataset[:,10]

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
parameters1 = [
   {'solver': ['svd', 'lsqr', 'eigen'], 
    'shrinkage': ['auto'],
    'store_covariance': [True, False]},
  ]

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=42)

def LDA(T, parameters_here):
    model = LinearDiscriminantAnalysis()

    gs = GridSearchCV(model, parameters_here, scoring = 'accuracy', cv=T, n_jobs=5)

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
T = 5 # Número de folds

print("\n==== LDA ====")
LDA(T, parameters1)