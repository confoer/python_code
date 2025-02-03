from turtle import st
from matplotlib import axis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def Linear_Regression(X, w, b):
    """
        X: 二维 Numpy 数组，行数为数据数，列数为特征数
        w: X 的权重
        b: 方程偏差值
    """
    if X.shape[1] == 1:
        y = w * X + b
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"真实参数：斜率 w = {w}, 截距 b = {b}")
        print(f"模型估计的斜率：{model.coef_[0]:.2f}, 截距：{model.intercept_:.2f}")
        print(f"模型的均方误差 (MSE): {mse:.2f}")
        print(f"模型的决定系数 (R²): {r2:.2f}")
        
        plt.scatter(X_test, y_test, color='blue', label='实际值')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('简单线性回归')
        plt.legend()
        plt.show()
    else:
        y = np.dot(X, w) + b
        data = pd.DataFrame(X, columns=[f'Feature{i+1}' for i in range(X.shape[1])])
        data['Target'] = y
        data.dropna(inplace=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"模型的平均绝对误差 (MAE): {mae:.2f}")
        print(f"模型的均方误差 (MSE): {mse:.2f}")
        print(f"模型的决定系数 (R²): {r2:.2f}")
        print(f"模型的系数 (斜率): {model.coef_}")
        print(f"模型的截距: {model.intercept_:.2f}")
        
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

def Logistic_Regression(X,y):
    """
        X: 二维 Numpy 数组，行数为数据数，列数为特征数
        y:多分类标签
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler =StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"模型的准确率: {accuracy:.2f}")
    print("混淆矩阵:")
    print(conf_matrix)
    print("分类报告:")
    print(class_report)

    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        X_train_pca = pca.transform(X_train)
        model.fit(X_train_pca, y_train)

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    else:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
         
        
    h = .02  # 网格步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()