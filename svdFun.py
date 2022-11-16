from solveFun import np


def getRandomVector(n):
    mu, sigma = 0, 1
    unnormVector = np.random.normal(mu, sigma, n)
    normVector = unnormVector / np.linalg.norm(unnormVector)
    return normVector


def getSingularVector(A, epsilon=1e-15, maxIter=100):
    n, m = A.shape
    x = getRandomVector(min(n, m))
    lastV = None
    currentV = x

    B = np.dot(A.T, A) if (n > m) else np.dot(A, A.T)

    iter = 0
    while (iter < maxIter):
        iter += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / np.linalg.norm(currentV)

        if (abs(np.dot(currentV, lastV)) > 1 - epsilon):
            break

    return currentV


def getSVD(A, epsilon=1e-15):
    n, m = A.shape
    svdSoFar = []

    for i in range(min(n, m)):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = getSingularVector(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = np.linalg.norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = getSingularVector(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = np.linalg.norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs


def getAnsSVD(U, S, V, arrY):
    diagS = np.diag(S)
    invS = np.linalg.inv(diagS)
    transposeV = np.transpose(V)
    transposeU = np.transpose(U)
    ans = np.dot(transposeV, np.dot(invS, np.dot(transposeU, arrY)))

    return ans
