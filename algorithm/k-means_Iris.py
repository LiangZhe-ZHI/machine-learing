# k-means_Iris
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 载入数据
iris = load_iris()
# 检测样本
# print(iris.data.shape, iris.target.shape)
# print(iris.target_names)
# print(iris.feature_names)


# 格式化数据
pd.set_option('precision', 2)
pd.set_option('max_columns', 5)
pd.set_option('display.width', None)
# 创建DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 添加一个包含每个样品品种名称的列species
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# 可视化数据
# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# grid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4], hue='species')
# plt.show()


# 使用KMeans估计器
kmeans = KMeans(n_clusters=3, random_state=11)
kmeans.fit(iris.data)

# 使用PCA估计器进行降维
pca = PCA(n_components=2, random_state=11)
pca.fit(iris.data)
iris_pca = pca.transform(iris.data)

# 可视化降维数据
iris_pca_df = pd.DataFrame(iris_pca, columns=['Component1', 'Component2'])
iris_pca_df['species'] = iris_df.species
axes = sns.scatterplot(data=iris_pca_df, x='Component1', y='Component2',
                       hue='species', legend='brief', palette='cool')
# PCA降维质心
iris_centers = pca.transform(kmeans.cluster_centers_)
dots = plt.scatter(iris_centers[:, 0], iris_centers[:, 1], s=100, c='k')
plt.show()


# 选择最佳聚类分类器
estimators = {
    'KMeans': kmeans,
    'DBSCAN': DBSCAN(),
    'MeanShift': MeanShift(),
    'SpectralClustering': SpectralClustering(n_clusters=3),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3)
}

for name, estimator in estimators.items():
    estimator.fit(iris.data)
    print(f'\n{name:}')
    for i in range(0, 101, 50):
        labels, counts = np.unique(estimator.labels_[i:i+50], return_counts=True )
        print(f'{i}-{i+50}:')
        for label, count in zip(labels, counts):
            print(f'   label={label}, count={count}')


