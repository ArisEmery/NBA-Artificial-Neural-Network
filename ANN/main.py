#!/usr/bin/python

#####################################
# module: cs3430_s18_hw07.py
# Aris Emery
# A01984177
# Accuracy info:
# At 39 iterations, I can get 120/129 correct. Any less typically results in 64 or less correct.
# At 40+ I can get 127/129 correct
# somewhere between 50000 and 60000, it goes from 127 to 128
# 128/129 was my best accuracy as I stuck with 100000000 or fewer iterations
#####################################

import numpy as np
from numpy import genfromtxt
import pickle as cPickle

#from cs3430_s18_hw07_data import *

filename="/Users/arisemery/CS5665 work/project/JameHarden2018-19.csv"
filename2="/Users/arisemery/CS5665 work/project/LebronJames2018-19.csv"
my_data = genfromtxt(filename, delimiter=',')
my_data2=genfromtxt(filename2, delimiter=',')

# sigmoid function
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# takes a n-tuple of layer dimensions,
# i.e., the numbers of neurons in each layer of ANN and returns a (n − 1)-tuple of weight matrices
# initialized with random floats with a mean of 0 and a standard deviation of 1 for the corresponding
# n-layer ANN.
def build_nn_wmats(mat_dims):
    #mat_tuple=()
    mat_list=[]
    for x in range(0,len(mat_dims)-1):
        np.random.seed(1)
        new_mat=np.random.rand(mat_dims[x], mat_dims[x+1])
        for i in range(0, mat_dims[x]):
            for j in range(0, mat_dims[x+1]):
                new_mat[i][j]=new_mat[i][j]-.5
        #print(new_mat)
        mat_list.append(new_mat)
        #print(mat_dims[x])
    mat_tuple = tuple(mat_list)
    return mat_tuple

def build_even_odd_nn():
    return build_nn_wmats((10,15,15,2))

def build_231_nn():
    return build_nn_wmats((2, 3, 1))


def build_838_nn():
    return build_nn_wmats((8, 3, 8))

def build_949_nn():
    return build_nn_wmats((9, 4, 9))


def create_nn_data():
    mat_list = []
    for x in range(1, len(my_data)):
        new_mat = np.array((0.0,0,0,0,0,0,0,0,0,0))
        new_mat[0] = my_data[x][1]
        new_mat[1] = my_data[x][2]
        new_mat[2] = my_data[x][3]
        new_mat[3] = my_data[x][4]
        new_mat[4] = my_data[x][5]
        new_mat[5] = my_data[x][6]
        new_mat[6] = my_data[x][7]
        new_mat[7] = my_data[x][8]
        new_mat[8] = my_data[x][9]
        new_mat[9] = my_data[x][10]
        mat_list.append(new_mat)
    arraynp = np.array(mat_list)

    mat_list2 = []
    ##even is win, odd is loss
    for x in range(1, len(my_data)):
        win=my_data[x][11]
        if win==1:
            new_mat = np.array((1,0))
        else:
            new_mat = np.array((0,1))
        mat_list2.append(new_mat)
    arraynp2 = np.array(mat_list2)

    return arraynp, arraynp2

def create_nn_data_2():
    mat_list = []
    for x in range(1, len(my_data2)):
        new_mat = np.array((0.0,0,0,0,0,0,0,0,0,0))
        new_mat[0] = my_data2[x][1]
        new_mat[1] = my_data2[x][2]
        new_mat[2] = my_data2[x][3]
        new_mat[3] = my_data2[x][4]
        new_mat[4] = my_data2[x][5]
        new_mat[5] = my_data2[x][6]
        new_mat[6] = my_data2[x][7]
        new_mat[7] = my_data2[x][8]
        new_mat[8] = my_data2[x][9]
        new_mat[9] = my_data2[x][10]
        mat_list.append(new_mat)
    arraynp = np.array(mat_list)

    mat_list2 = []
    ##even is win, odd is loss
    for x in range(1, len(my_data2)):
        win=my_data2[x][11]
        if win==1:
            new_mat = np.array((1,0))
        else:
            new_mat = np.array((0,1))
        mat_list2.append(new_mat)
    arraynp2 = np.array(mat_list2)

    return arraynp, arraynp2


def train_4_layer_nn(numIters, X, y, build,learningRate):
    W1, W2, W3 = build()
    # print(X)
    for j in range(numIters):
        Z2 = np.dot(X, W1)
        a2 = sigmoid(Z2)

        Z3 = np.dot(a2, W2)
        a3 = sigmoid(Z3)

        Z4=np.dot(a3,W3)
        yHat = sigmoid(Z4)

        yHat_error = y - yHat
        yHat_delta = yHat_error * sigmoid(yHat, deriv=True)

        a3_error = yHat_delta.dot(W3.T)
        a3_delta = a3_error * sigmoid(a3, deriv=True)

        a2_error = a3_delta.dot(W2.T)
        a2_delta = a2_error * sigmoid(a2, deriv=True)

        tester=a3.T.dot(yHat_delta)

        W3 += a3.T.dot(yHat_delta)*(learningRate)
        W2 += a2.T.dot(a3_delta)*(learningRate)
        W1 += X.T.dot(a2_delta)*(learningRate)

    return W1, W2, W3


def fit_4_layer_nn(x, wmats, thresh=.4, thresh_flag=True):
    W1=wmats[0]
    W2=wmats[1]
    W3=wmats[2]
    a2 = sigmoid(np.dot(x, W1))
    a3 = sigmoid(np.dot(a2, W2))
    yHat = sigmoid(np.dot(a3, W3))
    if thresh_flag == True:
        for y in np.nditer(yHat, op_flags=['readwrite']):
            if y > thresh:
                y[...] = 1
            else:
                y[...] = 0
        return yHat.astype(float)
    else:
        return yHat


def is_win(n, wmats):
    X,y=create_nn_data()
    result = fit_4_layer_nn(X[n], wmats, thresh_flag=True)
    # print(result)
    if result[0]==1 and result[1]==0:
        return True

    else:
        return False


def eval_win_loss_nn(wmats):
    X, y = create_nn_data()
    #reslt = fit_4_layer_nn(X, wmats, thresh_flag=True)
    num_correct=0
    num_incorrect=0
    for x in range(0,len(X)):
        # z=is_win(x,wmats)
        # print(z)
        z=y[x][0]
        tota=y[x]
        if (y[x][0]==1) and (y[x][1]==0) and is_win(x,wmats)==True:
            num_correct+=1
        elif (y[x][0]==0) and (y[x][1]==1) and is_win(x,wmats)==False:
            num_correct+=1
        else:
            num_incorrect+=1
    print(num_incorrect)
    return num_correct

def count_nn_wins(wmats):
    X, y = create_nn_data()
    num_wins=0
    num_losses=0
    for x in range(0,len(X)):
        # z=is_win(x,wmats)
        # print(z)
        z=y[x][0]
        tota=y[x]
        if is_win(x,wmats)==True:
            num_wins+=1
        elif is_win(x,wmats)==False:
            num_losses+=1
    return num_wins, num_losses

def count_nn_wins_2(wmats):
    X, y = create_nn_data_2()
    num_wins=0
    num_losses=0
    for x in range(0,len(X)):
        # z=is_win(x,wmats)
        # print(z)
        z=y[x][0]
        tota=y[x]
        if is_win(x,wmats)==True:
            num_wins+=1
        elif is_win(x,wmats)==False:
            num_losses+=1
    return num_wins, num_losses




def main():
    ##todo rn we have 8 inputs, the binary reps of the numbers, and a 01 or 10 output
    ##todo out project needs 10 inputs rn, same number of outputs
    ##todo instead of is even, we will do is win
    X, y = create_nn_data() ##change this to 2 to switch player
    print(len(X))
    wmats = train_4_layer_nn(100000, X, y, build_even_odd_nn,.001)
    result=eval_win_loss_nn(wmats)
    print(result)
    wins, losses = count_nn_wins(wmats)
    print("stats on trained player:")
    print("wins:",wins)
    print("losses:",losses)
    print("wins %", (wins/(wins+losses)))
    print("stats on test player:")
    wins, losses = count_nn_wins_2(wmats)
    print("wins:", wins)
    print("losses:", losses)
    print("wins %", (wins/(wins+losses)))







    # my_data = genfromtxt(filename, delimiter=',')
    # print(len(my_data))
    # mat_list=[]
    # new_mat = np.array((0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    # new_mat[0] = my_data[1][1]
    # new_mat[1] = my_data[1][2]
    # new_mat[2] = my_data[1][3]
    # new_mat[3] = my_data[1][4]
    # new_mat[4] = my_data[1][5]
    # new_mat[5] = my_data[1][6]
    # new_mat[6] = my_data[1][7]
    # new_mat[7] = my_data[1][8]
    # new_mat[8] = my_data[1][9]
    # new_mat[9] = my_data[1][10]
    # mat_list.append(new_mat)
    #
    # print(new_mat)

    # wmats = build_even_odd_nn()
    # print(type(wmats[2]))
    # my_int=129
    # my_bin_int=np.binary_repr(my_int)
    # print(my_bin_int)
    # X,y=create_nn_data()
    # #print(x[:10])
    # even_odd_wmats = train_4_layer_nn(1000, X, y, build_even_odd_nn)
    # test=eval_even_odd_nn(even_odd_wmats)
    # print(test)
    # print(len(even_odd_wmats))
    # print(even_odd_wmats)



if __name__ == "__main__":
    main()



