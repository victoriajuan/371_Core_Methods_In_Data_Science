import numpy as np
import pandas as pd
import time

å = 0.85
e = 0.00001

def createAM(nodes, link):
    am = np.zeros((nodes, nodes))
    with open(link) as f:
        for i, col in enumerate(f):
            col = col.split(',')
            am[int(col[1])][int(col[0])] = int(col[2])
    return am

def digonalZero(am):
    np.fill_diagonal(am, 0)

def normalizeCol(am):
    # sum up each column 
    sum = am.sum(axis=0)
    # divide each entry in a column by the sum of that column
    return np.divide(am, sum, out=np.zeros_like(am), where=sum!=0)

def danglingNode(am):
    sum = am.sum(axis=0)
    # if the sum of each column is 0, it means there is no citation
    return (sum == 0).astype(float)

def articleVector(npMatrix):
    '''npMatrix is the total number of articles published by all of the journals'''
    # column vector of the number of articles published in each journal over the (five-year) target window, 
    # normalized so that its entries sum to 1
    return npMatrix / npMatrix.sum()

def initialVector(a, am):
    return np.ones_like(a) / am.shape[0]

def calPiK1(H, piK, danglingNode, a):
    # π(k+1) euqation
    p1 = (å * H).dot(piK)
    p2 = (å * danglingNode).dot(piK)
    p2 = p2  + (1 - å)
    p2 = np.multiply(p2, a)
    # return piK1
    return p1 + p2

def iteration(H, pi0, danglingNode, a):
    # initialize the piK1 and norm
    piK1 = calPiK1(H, pi0, danglingNode, a)
    norm = np.linalg.norm((piK1 - pi0))
    # first iteration piK is equal to pi0
    piK = pi0
    
    counter = 0
    # while residual is less than e, piK ~ piK1 is the influence vector
    while norm > e:
        # calculate the norm again and update the current influence vector to iterate
        piK1 = calPiK1(H, piK, danglingNode, a)
        norm = np.linalg.norm((piK1 - piK))
        piK = piK1
        counter += 1
    return counter, piK1

def calEFSco(H, pi):
    Hpi = H.dot(pi)
    return 100*(Hpi/Hpi.sum())


def main():
    start_time = time.time()
    am = createAM(10748, './links.txt')
    digonalZero(am)
    H = normalizeCol(am)
    danglingN = danglingNode(am)
    tempAM = np.empty((10748))
    tempAM[:] = 1
    articleVec = articleVector(tempAM)
    intialVec = initialVector(articleVec, am)
    counter, influenceVector = iteration(H, intialVec, danglingN, articleVec)
    eiganfactor = calEFSco(H, influenceVector)

    # print the top 20 journals of Eiganfactor score
    indexSort = eiganfactor.argsort()
    indexReversed = indexSort[::-1]
    top20 = indexReversed[:20]
    for journal in top20:
        print("{0}: {1}".format(journal, eiganfactor[journal]))

    # iterated 21 times
    print("Iteration time:{0}".format(counter))
    # ≈ 30 seconds
    print("Run time: {0} seconds.".format(time.time() - start_time))


if __name__ == "__main__":
    main()