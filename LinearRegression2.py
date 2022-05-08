

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]

x, y = np.array(x), np.array(y)

model = LinearRegression().fit(x, y)

print("截距：", model.coef_)
print("系数：", model.intercept_)
r_sq = model.score(x, y)
print('aaa:', r_sq)

y_pred = model.predict(x)

print('predicited response:', y_pred, sep='\n')

x_new = np.arange(10).reshape((-1, 2))
print('xnew:', x_new)

y_new = model.predict(x_new)
print('ynew:', y_new)
