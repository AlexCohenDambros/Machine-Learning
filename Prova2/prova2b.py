from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import preprocessing

import warnings
from warnings import simplefilter
warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)

random_state = 42
np.random.seed(42)

X, y = load_diabetes(return_X_y=True)

scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

# separa teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)


parameters_BaggingRegressor = [
    {'base_estimator': [None, DecisionTreeClassifier(criterion='entropy', max_depth=5), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=3, weights='distance')],
     'n_estimators': range(5, 16, 2),
     'max_features': [0.5, 0.7, 1],
     'random_state': [random_state],
     }
]

parameters_RandomForestRegressor = [
    {'n_estimators': [100*x for x in range(1, 5)],
     'criterion': ["squared_error", "absolute_error", "poisson"],
     'max_depth': [None]+list(range(5, 16, 2)),
     'random_state': [random_state],
     }
]


parameters_MLPRegressor = [
    {'solver': ['adam', 'sgd'],
     'hidden_layer_sizes': [4, 7, (6, 8)],
     'max_iter': [100, 300, 500, 800, 1300],
     }
]

parameters_SVR = [
    {'kernel': ["linear", "poly", "rbf", "sigmoid"],
     'gamma': ['scale', 'auto']
    }
]

# ================= Funções de aprendizagem - Regressores =================

def svr(parameters, folds):

    # Treina o Regressor
    clfa = SVR()

    clfa = GridSearchCV(clfa, parameters, scoring='r2', cv=folds,  n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    result = model_selection.cross_val_predict(clfa, X, y, cv=folds)
    mae = mean_absolute_error(y, result)
    
    print("\nConcluido!")
    
    return mae, best_parameters


def randomForestRegressor(parameters, folds):

    # Treina o Regressor
    clfa = RandomForestRegressor(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='r2', cv=folds,  n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    result = model_selection.cross_val_predict(clfa, X, y, cv=folds)
    mae = mean_absolute_error(y, result)
    
    print("\nConcluido!")
 
    return mae, best_parameters


def baggingRegressor(parameters, folds):

    # Treina o Regressor
    clfa = BaggingRegressor(random_state=42)

    clfa = GridSearchCV(
        clfa, parameters, scoring='r2', cv=folds,  n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    result = model_selection.cross_val_predict(clfa, X, y, cv=folds)
    mae = mean_absolute_error(y, result)

    print("\nConcluido!")

    return mae, best_parameters


def mlpRegressor(parameters, folds):

    # Treina o Regressor
    clfa = MLPRegressor()

    clfa = GridSearchCV(
        clfa, parameters, scoring='r2', cv=folds,  n_jobs=-1)

    clfa = clfa.fit(X_train, y_train)

    best_parameters = clfa.best_params_

    result = model_selection.cross_val_predict(clfa, X, y, cv=folds)
    mae = mean_absolute_error(y, result)

    print("\nConcluido!")
    
    return mae, best_parameters

# =========================================================================

def processing_algorithms_regressor():
    folds = 5

    print("\n==== Executando --> SVR ====")
    mae, best_parameters = svr(parameters_SVR, folds)
    melhoresResultados["SVR"] = [mae, best_parameters]

    print("\n==== Executando --> Random Forest Regressor ====")
    mae, best_parameters = randomForestRegressor(
        parameters_RandomForestRegressor, folds)
    melhoresResultados["RandomForestRegressor"] = [mae, best_parameters]

    print("\n==== Executando --> Bagging Regressor====")
    mae, best_parameters = baggingRegressor(parameters_BaggingRegressor, folds)
    melhoresResultados["BaggingRegressor"] = [mae, best_parameters]

    print("\n==== Executando --> MLP_Regressor ====")
    mae, best_parameters = mlpRegressor(parameters_MLPRegressor, folds)
    melhoresResultados["MLP_Regressor"] = [mae, best_parameters]

    print("================================")
    print("\nResultados encontrados: \n")
    melhorAcuracia = ["", 1000]
    for key in melhoresResultados:
        print("Mean Absolute Error {}: {}".format(
            key, melhoresResultados[key][0]))
        if melhorAcuracia[1] > melhoresResultados[key][0]:
            melhorAcuracia[0] = key
            melhorAcuracia[1] = melhoresResultados[key][0]

    print("================================")
    print("\nMelhor resultado foi do: {} \nMean Absolute Error: {}".format(
        melhorAcuracia[0], melhorAcuracia[1]))
    for key in melhoresResultados:
        if key == melhorAcuracia[0]:
            mae, best_parameters = melhoresResultados[key][
                0], melhoresResultados[key][1]

    print("\nParametros utilizados: {}".format(best_parameters))

# ===================================================================================

melhoresResultados = {}
print("\n ========== Questao 2 da prova ==========")
processing_algorithms_regressor()