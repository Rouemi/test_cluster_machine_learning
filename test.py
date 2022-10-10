import numpy as np
import pandas as pd
import csv
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# read csv file
df = pd.read_csv('data2.csv', sep=';', index_col='id')
df.describe()
scaler = StandardScaler()


df[['q1_T', 'q2_T', 'q3_T', 'q4_T', 'q5_T']] = scaler.fit_transform(df[['q1', 'q2', 'q3', 'q4', 'q5']])

def optimise_k_means(data, max_k) :
    means = []
    inertias = []

    for k in range(1, max_k) :
        kmeans = KMeans(n_clusters=k).fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    
    #plot
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

#optimise_k_means(df[['q1_T', 'q2_T', 'q3_T', 'q4_T', 'q5_T']], 20)

cluclu = KMeans(n_clusters=10).fit(df[['q1_T', 'q2_T', 'q3_T', 'q4_T', 'q5_T']])
df['cluster'] = cluclu.labels_

#Save to csv all id with cluster
file = open('test.csv', 'w')
# save id and cluster
for i in range(len(df)):
    file.write(str(df.index[i]) + ';' + str(df['cluster'][i]) + '\n')
file.close()




