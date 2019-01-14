from sklearn import linear_model
from matplotlib import pyplot as plt

def runplt():
   plt.figure()
   plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
   plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
   plt.title("线性回归实验")
   plt.xlabel("x")
   plt.ylabel("y")
   plt.axis([0, 25, 0, 25])
   plt.grid()
   return plt

if __name__ == "__main__":

    plt = runplt()

    X = [[0], [1], [2], [3], [4], [5]]
    y = [[0], [1], [3], [3], [3], [5]]

    X_test = [[6], [7], [8]]
    
    model = linear_model.LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X_test)
    print(y_pred)

    plt.scatter(X, y, color='r', marker='o')
    plt.scatter(X_test, y_pred,color='b', marker='+')
    plt.plot(X, y_pred)
    plt.show()