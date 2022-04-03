import itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

X, y = load_breast_cancer(return_X_y=True)


Criterion = ['gini', 'entropy']
Max_depth = range(1,10)
min_sample_split = range(2,10)
min_sample_leaf = range(1,10)
splitter= ['best', 'random']

valores = [Criterion, Max_depth, min_sample_split, min_sample_leaf, splitter]

lista = itertools.product(*valores)

melhorResultado = 0

for valor in lista:

    
    # Definindo a técnica a ser utilizada
    clf = DecisionTreeClassifier(criterion=valor[0], max_depth=valor[1], min_samples_split=valor[2], min_samples_leaf=valor[3], splitter=valor[4], random_state=42)

    # Usando a validação cruzada com 10 folds neste exemplo.
    T=5 # número de pastas ou folds
    result = model_selection.cross_val_score(clf, X, y, cv=T)

    if result.mean() > melhorResultado:
    
        print(valor)
        melhorResultado = result.mean()

        # Mostrando a acurácia média e desvio padrão.
        print("\nCross Validation Results %d folds:" % T)
        print("Mean Accuracy: %.5f" % result.mean())
        print("Mean Std: %.5f" % result.std())

        # Calculando a predição para exemplo de teste
        y_pred = model_selection.cross_val_predict(clf, X, y, cv=T)

        # Calculando para cada instância de teste a probabilidade de cada classe
        predicted_proba=model_selection.cross_val_predict(clf, X, y, cv=T, method='predict_proba')

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

print('Done')