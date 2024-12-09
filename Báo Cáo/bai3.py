import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import pi
import argparse

def perform_kmeans(df, columns_to_cluster, num_clusters=4):
    data = df[columns_to_cluster].replace("N/A", 0).astype(float)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=num_clusters)
    df['Cluster'] = kmeans.fit_predict(data_scaled)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    df['PCA1'] = data_pca[:, 0]
    df['PCA2'] = data_pca[:, 1]

    plt.figure(figsize=(10, 8))
    for cluster in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.title('K-means Clustering with PCA')
    plt.show()
    return df

def plot_radar(df, player1, player2, attributes):
    attributes = attributes.split(",")
    data = df[df['Player'].isin([player1, player2])]
    data = data.set_index('Player')[attributes].astype(float)

    values1 = data.loc[player1].values.flatten().tolist()
    values2 = data.loc[player2].values.flatten().tolist()
    values1 += values1[:1]
    values2 += values2[:1]

    angles = [n / float(len(attributes)) * 2 * pi for n in range(len(attributes))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values1, linewidth=1, linestyle='solid', label=player1)
    ax.fill(angles, values1, alpha=0.3)
    ax.plot(angles, values2, linewidth=1, linestyle='solid', label=player2)
    ax.fill(angles, values2, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("results.csv")
    columns_to_cluster = ['Min', 'Gls', 'Ast', 'xG', 'xA', 'Shots', 'Cmp%', 'Tkl', 'Int', 'PrgC']
    df = perform_kmeans(df, columns_to_cluster)

    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", required=True)
    parser.add_argument("--p2", required=True)
    parser.add_argument("--Attribute", required=True)
    args = parser.parse_args()
    plot_radar(df, args.p1, args.p2, args.Attribute)
