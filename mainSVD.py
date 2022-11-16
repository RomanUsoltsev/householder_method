import solveFun as solf
from solveFun import np
import secondaryFun as secf
from secondaryFun import pd
from secondaryFun import plt

import svdFun as svf


def mainSVDTab(showplt=True):

    globN = 11

    for N in range(globN):
        # load data
        X3, Y3, XY3 = secf.getData('./data/data_3.txt')
        X4, Y4, XY4 = secf.getData('./data/data_4.txt')
        X5, Y5, XY5 = secf.getData('./data/data_5.txt')

        # get SLAY
        arr3 = solf.getSLAY(X3, N)
        arr4 = solf.getSLAY(X4, N)
        arr5 = solf.getSLAY(X5, N)

        # get My SVD
        S3, U3, V3 = svf.getSVD(arr3)
        S4, U4, V4 = svf.getSVD(arr4)
        S5, U5, V5 = svf.getSVD(arr5)

        # get My ans SVD
        ansX3 = svf.getAnsSVD(U3, S3, V3, Y3)
        ansX4 = svf.getAnsSVD(U4, S4, V4, Y4)
        ansX5 = svf.getAnsSVD(U5, S5, V5, Y5)

        # get Standart SVD
        U3_ch, S3_ch, V3_ch = np.linalg.svd(arr3)
        U4_ch, S4_ch, V4_ch = np.linalg.svd(arr4)
        U5_ch, S5_ch, V5_ch = np.linalg.svd(arr5)

        # get Standart ans SVD
        ansX3_ch = svf.getAnsSVD(U3_ch[:, :N + 1], S3_ch, V3_ch, Y3)
        ansX4_ch = svf.getAnsSVD(U4_ch[:, :N + 1], S4_ch, V4_ch, Y4)
        ansX5_ch = svf.getAnsSVD(U5_ch[:, :N + 1], S5_ch, V5_ch, Y5)

        # show SVD solve and standart
        print("solve data3 [my] , [standart], |my - standatr| :\n", np.c_[ansX3, ansX3_ch, ansX3 - ansX3_ch])

        print("solve data4 [my] - [standart], |my - standatr| :\n", np.c_[ansX4, ansX4_ch, ansX4 - ansX4_ch])

        print("solve data5 [my] - [standart], |my - standatr| :\n", np.c_[ansX5, ansX5_ch, ansX5 - ansX5_ch])

        # get New Y data
        arrNewY3 = secf.getFunValueArr(X3, ansX3)
        arrNewY4 = secf.getFunValueArr(X4, ansX4)
        arrNewY5 = secf.getFunValueArr(X5, ansX5)

        if (showplt):
            plt.figure(figsize=(35, 8), facecolor='white')

            secf.showData(1, 3, 1, X3, arrNewY3, "my", "N = " + str(N))
            secf.showData(1, 3, 1, X3, ansX3_ch, "orig", "N = " + str(N))

            secf.showData(1, 3, 2, X4, arrNewY4, "my", "N = " + str(N))
            secf.showData(1, 3, 2, X4, ansX3_ch, "orig", "N = " + str(N))

            secf.showData(1, 3, 3, X5, arrNewY5, "my", "N = " + str(N))
            secf.showData(1, 3, 3, X5, ansX3_ch, "orig", "N = " + str(N))

            plt.show()

        # show SME
        print("SME 3 :", secf.getSME(Y3, arrNewY3))
        print("SME 4 :", secf.getSME(Y4, arrNewY4))
        print("SME 5 :", secf.getSME(Y5, arrNewY5))
