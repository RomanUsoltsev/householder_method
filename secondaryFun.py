import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from solveFun import polyCheb
from solveFun import np


def getData(path):
    df = pd.read_csv(path, sep=" ", dtype=float)
    X = np.array(df.iloc[:, 0].values)
    Y = np.array(df.iloc[:, 1].values)
    XY = np.array(np.c_[X, Y])
    return X, Y, XY


def showData(a, b, index, X, Y, lb, tl):
    splot = plt.subplot(a, b, index)
    plt.plot(X, Y, label=lb)
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(tl)
    plt.legend()
    return splot


def getFunValue(arrAnsX, xPoint):
    arr = np.zeros(arrAnsX.shape)
    for i in range(arr.shape[0]):
        arr[i] = polyCheb(xPoint, i)
    return np.sum(np.multiply(arr, arrAnsX))


def getFunValueArr(xArray, arrAnsX):
    getNewY = np.vectorize(lambda x: getFunValue(arrAnsX, x))
    arrYNew = getNewY(xArray)

    return arrYNew


def getSME(Y, newY):
    resArr = np.subtract(Y, newY)
    resArr = np.multiply(resArr, resArr)
    resSum = ((np.sum(resArr) / Y.shape[0]) ** (1 / 2)) / (np.max(Y))

    return resSum
