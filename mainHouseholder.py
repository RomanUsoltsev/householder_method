import solveFun as solf
from solveFun import np
import secondaryFun as secf
from secondaryFun import pd
from secondaryFun import plt


def mainTab(showplt = True):
    # load data
    X3, Y3, XY3 = secf.getData('./data/data_3.txt')
    X4, Y4, XY4 = secf.getData('./data/data_4.txt')
    X5, Y5, XY5 = secf.getData('./data/data_5.txt')

    # get SLAY
    globN = 11
    tabArr3 = np.zeros([globN, 5])
    tabArr4 = np.zeros([globN, 5])
    tabArr5 = np.zeros([globN, 5])
    N3 = np.array(range(globN))
    N4 = np.array(range(globN))
    N5 = np.array(range(globN))

    for i in range(globN):
        arr3 = solf.getSLAY(X3, N3[i])
        arr4 = solf.getSLAY(X4, N4[i])
        arr5 = solf.getSLAY(X5, N5[i])

        # transformHouseholder
        ans3, arrB3 = solf.transformHouseholder(arr3, Y3)
        ans4, arrB4 = solf.transformHouseholder(arr4, Y4)
        ans5, arrB5 = solf.transformHouseholder(arr5, Y5)

        # gauss Reverse
        arrAnsX3 = solf.gaussReverse(ans3, arrB3)
        arrAnsX4 = solf.gaussReverse(ans4, arrB4)
        arrAnsX5 = solf.gaussReverse(ans5, arrB5)

        # standart solve
        ansNE3 = np.linalg.lstsq(arr3, Y3, rcond=None)
        ansNE4 = np.linalg.lstsq(arr4, Y4, rcond=None)
        ansNE5 = np.linalg.lstsq(arr5, Y5, rcond=None)

        ansNE3 = np.reshape(ansNE3[0], [ansNE3[0].shape[0], 1])
        ansNE4 = np.reshape(ansNE4[0], [ansNE4[0].shape[0], 1])
        ansNE5 = np.reshape(ansNE5[0], [ansNE5[0].shape[0], 1])

        # show cond Number
        arrNE3_1 = np.dot(np.transpose(arr3), arr3)
        arrNE4_1 = np.dot(np.transpose(arr4), arr4)
        arrNE5_1 = np.dot(np.transpose(arr5), arr5)

        cond_AT_A3 = np.linalg.cond(arrNE3_1)
        cond_A3 = np.linalg.cond(arr3)

        cond_AT_A4 = np.linalg.cond(arrNE4_1)
        cond_A4 = np.linalg.cond(arr4)

        cond_AT_A5 = np.linalg.cond(arrNE5_1)
        cond_A5 = np.linalg.cond(arr5)

        # get New Y data
        arrNewY3 = secf.getFunValueArr(X3, arrAnsX3)
        arrNewY4 = secf.getFunValueArr(X4, arrAnsX4)
        arrNewY5 = secf.getFunValueArr(X5, arrAnsX5)

        arrNewY3_ch = secf.getFunValueArr(X3, ansNE3)
        arrNewY4_ch = secf.getFunValueArr(X4, ansNE4)
        arrNewY5_ch = secf.getFunValueArr(X5, ansNE5)

        if (showplt) :
            plt.figure(figsize=(35, 8), facecolor='white')

            secf.showData(1, 3, 1, X3, arrNewY3, "my", "N = " + str(i))
            secf.showData(1, 3, 1, X3, arrNewY3_ch, "orig", "N = " + str(i))

            secf.showData(1, 3, 2, X4, arrNewY4, "my", "N = " + str(i))
            secf.showData(1, 3, 2, X4, arrNewY4_ch, "orig", "N = " + str(i))

            secf.showData(1, 3, 3, X5, arrNewY5, "my", "N = " + str(i))
            secf.showData(1, 3, 3, X5, arrNewY5_ch, "orig", "N = " + str(i))

            plt.show()

        SME_NE3 = secf.getSME(Y3, arrNewY3_ch)
        SME_QR3 = secf.getSME(Y3, arrNewY3)

        SME_NE4 = secf.getSME(Y4, arrNewY4_ch)
        SME_QR4 = secf.getSME(Y4, arrNewY4)

        SME_NE5 = secf.getSME(Y5, arrNewY5_ch)
        SME_QR5 = secf.getSME(Y5, arrNewY5)

        tabArr3[i][0] = i
        tabArr3[i][1] = cond_AT_A3
        tabArr3[i][2] = SME_NE3
        tabArr3[i][3] = cond_A3
        tabArr3[i][4] = SME_QR3

        tabArr4[i][0] = i
        tabArr4[i][1] = cond_AT_A4
        tabArr4[i][2] = SME_NE4
        tabArr4[i][3] = cond_A4
        tabArr4[i][4] = SME_QR4

        tabArr5[i][0] = i
        tabArr5[i][1] = cond_AT_A5
        tabArr5[i][2] = SME_NE5
        tabArr5[i][3] = cond_A5
        tabArr5[i][4] = SME_QR5
        print("Number Iter : ", i)



    return tabArr3, tabArr4, tabArr5


if __name__ == "__main__":
    tab3, tab4, tab5 = mainTab(showplt = False)

    tab_pd3 = pd.DataFrame(tab3, columns=["N", "cond(A.T * A)", "SME(NE)", "cond(A)", "SME(QR)"])
    tab_pd4 = pd.DataFrame(tab4, columns=["N", "cond(A.T * A)", "SME(NE)", "cond(A)", "SME(QR)"])
    tab_pd5 = pd.DataFrame(tab5, columns=["N", "cond(A.T * A)", "SME(NE)", "cond(A)", "SME(QR)"])

    print("DATA 3")
    print(tab_pd3)

    print("\nDATA 4")
    print(tab_pd4)

    print("\nDATA 5")
    print(tab_pd5)