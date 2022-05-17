# Exercício: Aplicação de Agrupamento
# Exercícios Agrupamento (Clustering)

# Dado o arquivo de dados s1.csv contendo 5.000 instâncias (amostras) não supervisionadas (sem rótulos) cada uma composta de 2 atributos numéricos, o seu desafio é construir um modelo descritivo com o intuito de segmentar estes dados em padrões similares. Após descobrir grupos de dados similares, utilize-os como futuras classes e treine uma Máquina de Vetor de Suporte.

# Apresente:

# Qual a quantidade de grupos mais adequada para os dados em s1.csv?
# Qual o índice silhueta do seu melhor agrupamento?
# Qual a taxa de acerto do SVM treinado considerando holdout 70/30?
# Qual a taxa de acerto do SVM treinado considerando validação cruzada?
# Link para a base: s1.csv

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# probability = true -> usado quando quero uma probabilidade na saida de uma classe 
# random_state 42 
from sklearn.svm import SVC
import urllib
import urllib.request as request
from turtle import shape
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, train_test_split

database = pd.read_csv("AprendizagemNaoSupervisionada\s1.csv")

X =  database.to_numpy()

range_n_clusters = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25]
dt_iner = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    a = np.asarray(cluster_labels)
    np.savetxt('AprendizagemNaoSupervisionada/classes/classes'+str(n_clusters), a, fmt='%i', delimiter="\n")

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    print("For n_clusters =", n_clusters,
          "The inertia is :", clusterer.inertia_)
    dt_iner.append(clusterer.inertia_)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


import matplotlib.pyplot as plt

plt.plot(range_n_clusters,dt_iner)
plt.title('Elbow plot')
plt.xlabel('# cluster')
plt.ylabel('Inertia')
plt.show()


np.random.seed(42)


y=np.loadtxt("AprendizagemNaoSupervisionada/classes/classes15")


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
