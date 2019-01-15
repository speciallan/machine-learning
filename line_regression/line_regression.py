from sklearn import linear_model
from matplotlib import pyplot as plt
import numpy as np

def runplt():
   plt.figure()
   plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
   plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
   plt.title(u"LineRegression")
   plt.xlabel("x")
   plt.ylabel("y")
   plt.axis([0, 10, 0, 10])
   plt.grid()

   return plt

if __name__ == "__main__":

   # from matplotlib.font_manager import _rebuild
   # _rebuild()

   plt = runplt()

   X = [[0], [1], [2], [3], [4], [5], [10]]
   y = [[0], [1], [3], [3], [3], [5], [10]]

   X_test = [[6], [7], [8]]
    
   # model = linear_model.LinearRegression()
   model = linear_model.Ridge()
   model.fit(X, y)

   y_pred = model.predict(X_test)

   y_hat = model.predict(X)
   print(y_pred)
   print("回归系数:%s, 偏置项:%.2f" % (model.coef_, model.intercept_))
   print("损失函数:%.2f" % np.mean((model.predict(X) - y) ** 2))
   print("预测性能:%.2f" % model.score(X, y))

   plt.scatter(X, y, color='r', marker='o')
   plt.scatter(X_test, y_pred, color='g', marker='+', s=100)
   plt.plot(X, y_hat)
   plt.show()