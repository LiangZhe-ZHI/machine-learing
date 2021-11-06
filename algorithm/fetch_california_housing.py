# fetch_california_housing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 加载数据集
california = fetch_california_housing()
# 数据格式化的一些操作
pd.set_option('precision', 4)  # 精度设置
pd.set_option('max_columns', 9)  # 最大显示列（8个特征和针对目标房价中值添加的一列（california.target））
pd.set_option('display.width', None)  # 指定字符宽度


# 创建DataFrame
california_df = pd.DataFrame(california.data, columns=california.feature_names)
# california.target中存储的中值房价添加一列
california_df['MedHouseValue'] = pd.Series(california.target)
# print(california_df.head())


# 随机取样绘图
sample_df = california_df.sample(frac=0.1, random_state=17)
sns.set(font_scale=2)
sns.set_style('whitegrid')
for feature in california.feature_names:
    plt.figure(figsize=(16, 9))
    sns.scatterplot(data=sample_df, x=feature, y='MedHouseValue',
                    hue='MedHouseValue', palette='cool', legend=False)
# plt.show()


# 拆分数据进行训练和测试
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target,
                                                    random_state=11)
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)
# 为每个特征生成单独的回归系数和截距
# for i, name in enumerate(california.feature_names):
#     print(f'{name:>10}: {linear_regression.coef_[i]}')
# print(linear_regression.intercept_)
predicted = linear_regression.predict(X_test)
expected = y_test


# 可视化预测房价和期望房价
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)
figure2 = plt.figure(figsize=(9, 9))
axes = sns.scatterplot(data=df, x='Expected', y='Predicted', hue='Predicted',
                       palette='cool', legend=False)
start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
axes.set_xlim(start, end)
axes.set_ylim(start, end)
line = plt.plot([start, end], [start, end], 'k--')
plt.show()