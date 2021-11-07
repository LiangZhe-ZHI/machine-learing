# k-NN_Digits
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# 加载数据集
digits = load_digits()
# # 创建图像
# figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
# # 显示每个图像
# for item in zip(axes.ravel(), digits.images, digits.target):
#     axes, image, target = item
#     axes.imshow(image, cmap=plt.cm.gray_r)
#     axes.set_xticks([])  # 删除x轴
#     axes.set_yticks([])  # 删除y轴
#     axes.set_title(target)
# plt.tight_layout()

# 将数据划分为训练集和测试集，X表示样本，y表示目标值
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)
# 将数据加载到估计器中
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

# 开始预测
predicted = knn.predict(X=X_test)
expected = y_test
print(f'{predicted[:20]}\n', expected[:20])

# 所有与期望不符预测
print('所有与期望不符预测')
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print(wrong)

# 模型标准性指标
confusion = confusion_matrix(y_true=expected, y_pred=predicted)
names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, predicted, target_names=names))

# 可视化混淆矩阵
confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))
# heatmap为热图，annot为右侧颜色表，cmap指示所显示的颜色
axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')
plt.show()

# 使用Digits数据集和之前创建的KNeighborsClassifier估计器进行k折交叉验证
# 创建KFold对象
kfold = KFold(n_splits=10, random_state=11, shuffle=True)
# 使用cross_val_score函数训练和测试模型
scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)
print(scores)
print(f'Mean accuracy: {scores.mean():.2%}')
print(f'Accuracy standard deviation: {scores.std():.2%}')
print()

# 通过KNeighborsClassifier、SVC和GaussianNB模型分别进行训练，并进行比较
estimators = {
    'KNeighborsClassifier': knn,
    'SVC': SVC(gamma='scale'),
    'GaussianNB': GaussianNB()
}
# 运行
for estimator_name, estimator_object in estimators.items():
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    scores = cross_val_score(estimator=estimator_object, X=digits.data, y=digits.target, cv=kfold)
    print(f'{estimator_name:>20}' +
          f'mean accuracy={scores.mean():.2%}' +
          f'standard deviation={scores.std():.2%}')
print()

# 超参数调整
for k in range(1, 20, 2):
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)
    print(f'k={k:<2}; mean accuracy={scores.mean():.2%};' +
          f'standard deviation={scores.std():.2%}')
