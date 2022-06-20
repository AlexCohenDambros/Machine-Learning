# Importa bibliotecas necessárias 
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_diabetes
from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

np.random.seed = 42

# Neste exemplo a base de dados diabetes é composta por 442 instâncias (N=442), 
# e cada instância é representada por um vetor de 10 atributos (M=10).

X, y = load_diabetes(return_X_y=True)
print("Formato de X: ", X.shape)
print("Formato de y: ", y.shape)

# Treina do modelo com busca dos parâmetros via GridSearch
# Definição dos parâmetros a serem avaliados 

parameters = [
  {'max_depth': [3, 5, 10, 20], 'min_samples_split': [3, 5, 10], 'criterion':['absolute_error', 'squared_error'], 'splitter':['best', 'random']},
  ]

folds=5
model = DecisionTreeRegressor(random_state=42)

# separando uma parte para base de validação (20%)
X, X_val, y, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

# GridSearch para customizar os parâmetros sobre base de validação
gs = GridSearchCV(model, parameters, scoring = 'r2', cv=folds, n_jobs=5)
gs.fit(X_val, y_val)

df=gs.cv_results_
print(tabulate(df, headers='keys', tablefmt='psql'))
print("Melhores parâmetros encontrados: ", gs.best_params_)

# Definindo a técnica a ser utilizada
model=gs.best_estimator_

# Usando a validação cruzada com 5 folds neste exemplo.
T=5 # número de pastas ou folds
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
