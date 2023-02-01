# imports
import matplotlib.pyplot as plt
import numpy as np
from fcmeans import FCM
from scipy.stats import multivariate_normal as mvn
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA

cat = ['0', 'Film & Animation', 'Autos & Vehicles', '3', '4', '5', '6', '7', '8', '9', 'Music', '11', '12', '13',
       '14',
       'Pets & Animals', '16', 'Sports', 'Short Movies', 'Travel & Events', 'Gaming', 'Videoblogging',
       'People & Blogs',
       'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style', 'Education', 'Science & Technology',
       'Nonprofits & Activism', 'Movies', 'Anime/Animation', 'Action/Adventure', 'Classics', 'Comedy',
       'Documentary', 'Drama',
       'Family', 'Foreign', 'Horror', 'Sci-Fi/Fantasy', 'Thriller', 'Shorts', 'Shows', 'Trailers']


def calc_kMeans(df, n_clusters):
    """ returns Kmeans labels, cluster centers and Davies Bouldin score """

    x = df[['views', 'likes']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
    y_kmeans = kmeans.fit_predict(x)
    return y_kmeans, kmeans.cluster_centers_, metrics.davies_bouldin_score(df[['views', 'likes']], y_kmeans)


def plot_kMeans(df):
    """ Plot kmeans clustering of views and likes """
    plt.figure()
    plt.scatter(df['views'], df['likes'], s=10)
    plt.title(df.name, fontsize=18)
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.figure()

    x = df[['views', 'likes']]
    y_kmeans, centers, evaluation = calc_kMeans(df, 3)

    # plot the 3 clusters
    plt.scatter(
        (x['views'])[y_kmeans == 0], (x['likes'])[y_kmeans == 0],
        s=10, c='blue', label='Cluster 1'
    )

    plt.scatter(
        (x['views'])[y_kmeans == 1], (x['likes'])[y_kmeans == 1],
        s=10, c='red', label='Cluster 2'
    )

    plt.scatter(
        (x['views'])[y_kmeans == 2], (x['likes'])[y_kmeans == 2],
        s=10, c='green', label='Cluster 3'
    )

    # plot the centroids
    plt.scatter(
        centers[:, 0], centers[:, 1],
        s=200, alpha=0.6, c='black', label='Centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(df.name + " K-Means clustered", fontsize=18)
    print("Davies Bouldin Score: " + str(evaluation))


def calc_gmm(df, n_clusters):
    """ returns GMM labels, cluster centers and Davies Bouldin score """

    x = df[['views', 'likes']]
    x = x.to_numpy()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(x)
    prediction_gmm = gmm.predict(x)
    centers = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(x)
        centers[i, :] = x[np.argmax(density)]
    return prediction_gmm, centers, metrics.davies_bouldin_score(df[['views', 'likes']], prediction_gmm)


def plot_gmm(df):
    """ Plot gmm clustering of views and likes """
    plt.figure()
    plt.scatter(df['views'], df['likes'], s=10)
    plt.title(df.name, fontsize=18)
    plt.xlabel("Views")
    plt.ylabel("Likes")
    num_components = 5
    x = df[['views', 'likes']]
    x = x.to_numpy()
    prediction_gmm, centers, evaluation = calc_gmm(df, num_components)
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=prediction_gmm, s=25, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6, label='Centroids')
    plt.legend(scatterpoints=1)
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(df.name + " GMM clustered", fontsize=18)

    print("Davies Bouldin Score: " + str(evaluation))


def calc_dbscan(df, _):
    x = np.column_stack([df['views'][0:10000], df['likes'][0:10000]])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    db_scan = DBSCAN(eps=0.15, min_samples=10)
    clusters = db_scan.fit_predict(x_scaled)

    return clusters, 2, metrics.davies_bouldin_score(x_scaled, clusters)


def plot_dbscan(df):
    """ plot dbscan clustering of views and likes """
    plt.figure()
    plt.scatter(df['views'], df['likes'], s=10)
    plt.title(df.name, fontsize=18)
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.figure()

    x = np.column_stack([df['views'][0:10000], df['likes'][0:10000]])
    clusters, _, evaluation = calc_dbscan(df, -1)
    # plot the clusters
    plt.scatter(x[:, 0], x[:, 1], c=clusters, s=10, cmap="plasma")
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(df.name + " DBScan clustered", fontsize=18)
    print("Davies Bouldin Score: " + str(evaluation))


def calc_fcm(df, n_clusters):
    """ returns FCM labels, cluster centers and Davies Bouldin score """
    x = df[['views', 'likes']]
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(x)

    fcm_centers = fcm.centers
    fcm_labels = fcm.u.argmax(axis=1)
    return fcm_labels, fcm_centers, metrics.davies_bouldin_score(x, fcm_labels)


def plot_fcm(df):
    """ plot fuzzy c means clustering of views and likes """
    plt.figure()

    fcm_labels, fcm_centers, evaluation = calc_fcm(df, 3)
    cmap = ['red', 'blue', 'green']
    cmap = [cmap[label] for label in fcm_labels]
    plt.scatter(df['views'], df['likes'], s=10)
    plt.title(df.name, fontsize=18)
    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.figure()

    plt.scatter(df['views'], df['likes'], c=cmap, s=10, label=fcm_labels)
    plt.scatter(fcm_centers['views'], fcm_centers['likes'], c='black', s=200, alpha=0.6)

    plt.xlabel("Views")
    plt.ylabel("Likes")
    plt.title(df.name + " Fuzzy C-means clustered", fontsize=18)

    print("Davies Bouldin Score: " + str(evaluation))


def plot_PCA(df):
    """ plot PCA of views, likes, dislikes and comment count """
    x = df[['dislikes', 'views', 'likes', 'comment_count']]
    y = df['category_id']  # label
    labels = [str(num) + " " + cat[num] for num in y.to_numpy()]
    pca = PCA(n_components=1)
    projected_x = pca.fit_transform(x)
    result = pd.DataFrame(projected_x[:, 0], columns=['PCA'])
    result['Categories'] = y
    result['Color'] = pd.DataFrame(data=labels)
    sns.lmplot('PCA', 'Categories', data=result, fit_reg=False, scatter_kws={"s": 30}, hue='Color')
    plt.title(df.name + " PCA", fontsize=18)


def plot_ICA(df):
    """ plot ICA of views, likes, dislikes and comment count """
    x = df[['views', 'likes', 'dislikes', 'comment_count']]
    y = df['category_id']  # label
    labels = [str(num) + " " + cat[num] for num in y.to_numpy()]
    ica = FastICA(n_components=1)
    projected_x = ica.fit_transform(x)
    result = pd.DataFrame(abs(projected_x[:, 0]), columns=['ICA'])
    result['Categories'] = y
    result['Color'] = pd.DataFrame(data=labels)
    sns.lmplot('ICA', 'Categories', data=result, fit_reg=False, scatter_kws={"s": 30}, hue='Color')
    plt.title(df.name + " PCA", fontsize=18)


def plot_KPCA(df):
    """ plot KPCA of views, likes, dislikes and comment count """
    x = np.column_stack([df['views'][0:5000], df['likes'][0:5000], df['dislikes'][0:5000], df['comment_count'][0:5000]])
    y = df['category_id']  # label
    labels = [str(num) + " " + cat[num] for num in y.to_numpy()]
    kpca = KernelPCA(n_components=1, kernel='linear')  # kernel = linear / kernel = rbf
    projected_x = kpca.fit_transform(x)
    result = pd.DataFrame(projected_x[:, 0], columns=['KPCA'])
    result['Categories'] = y
    result['Color'] = pd.DataFrame(data=labels)
    sns.lmplot('KPCA', 'Categories', data=result, fit_reg=False, scatter_kws={"s": 30}, hue='Color')
    plt.title(df.name + " PCA", fontsize=18)


def plot_cluster_scores(function, df, domain):
    """ Plot a line of the Davies Bouldin Score relative to number of clusters in the given 'function' algorithm """
    scores = []
    for i in domain:
        scores.append(function(df, i)[2])

    plt.scatter(domain, scores)
    plt.plot(domain, scores, label=df.name)
    plt.legend()
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies Bouldin Score")
