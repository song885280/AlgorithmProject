'''
2022.4.27
CedarXu
'''

import matplotlib.pyplot as plt
from sklearn import linear_model

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [20]]

model = linear_model.LinearRegression()
model.fit(X,y)

print("截距：", model.coef_)
print("系数：", model.intercept_)

y_pred = model.predict(X)

plt.scatter(X, y)
plt.grid
plt.plot(X, y_pred, "r--")

plt.show()

result = model.predict([[12]])
print("直径为12的披萨价格为：", result)
