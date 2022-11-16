import numpy as np


def pointValueCheb(x, N):
    if (N == 0):
        return 1.0
    if (N == 1):
        return x
    if (N > 1):
        return 2.0 * x * pointValueCheb(x, N - 1) - pointValueCheb(x, N - 2)


def polyCheb(X, N):
    getVal = lambda x: pointValueCheb(x, N)
    return getVal(X)


def getSLAY(X, N):
    arr = np.zeros([N + 1, X.shape[0]])
    for i in range(arr.shape[0]):
        arr[i] = polyCheb(X, i)
    return np.transpose(arr)


def getCondArr(X, N):
    arr = np.zeros(N)
    for i in range(N):
        arr[i] = np.linalg.cond(getSLAY(X, i))
    return arr


def vectorU(X):
    eVector = np.zeros([1, X.shape[0]])
    eVector[0][0] = 1
    if (np.sign(X[0]) == 1):
        tildaU = np.add(X, np.linalg.norm(X) * eVector)
    else:
        tildaU = np.subtract(X, np.linalg.norm(X) * eVector)

    uVector = np.divide(tildaU, np.linalg.norm(tildaU))
    return np.transpose(uVector)


def transformHouseholder(arrSLAY, arrY):
    ans = np.copy(arrSLAY)
    arrB = np.copy(arrY)
    n = ans.shape[0]
    for j in range(ans.shape[1]):
        U = vectorU(ans[j: n, j])

        ans[j:n, j:n] = np.subtract(ans[j:n, j:n],
                                    2 * np.dot(U,
                                               np.dot(np.transpose(U), ans[j:n, j:n])))

        arrB[j:n] = np.subtract(arrB[j:n],
                                2 * np.dot(U,
                                           np.dot(np.transpose(U), arrB[j:n])))

    checkZero = np.vectorize(lambda x: 0.0 if (abs(x) < 1e-15) else x)
    return checkZero(ans), checkZero(arrB)


def gaussReverse(arrSLAY, arrY):
    arrX = np.zeros([arrSLAY.shape[1]])
    n = arrSLAY.shape[0]
    j = arrSLAY.shape[1]
    if (abs(arrSLAY[j - 1, j - 1]) > 1e-15):
        arrX[j - 1] = arrY[j - 1] / arrSLAY[j - 1, j - 1]
    else:
        arrX[j - 1] = 0.0
    for i in range(1, j):
        multVector = np.multiply(arrSLAY[j - i - 1, j - i:j], arrX[j - i:j + 1])
        divideEl = arrSLAY[j - i - 1, j - i - 1]

        if (abs(divideEl) > 1e-15):
            arrX[j - 1 - i] = (arrY[j - i - 1] - np.sum(multVector)) / divideEl
        else:
            arrX[j - 1] = 0.0

    return np.reshape(arrX, [arrX.shape[0], 1])
