from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

digits = load_digits()
figure, axes = plt.subplots(nrows=5, ncols=7, figsize=(7, 5))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test
print(f'{predicted[:20]}\n', expected[:20])

print('所有与期望不符预测')
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print(wrong)