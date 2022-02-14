#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Homework 1: Question 3

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing packages
import cv2
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def read_data(CSVFile, Cols):
    df = pd.read_csv(CSVFile, usecols=Cols)
    return df.to_numpy().transpose()


def compute_covariance_matrix(data):

    means = [np.mean(x) for x in data]

    C = np.empty([len(data), len(data)])
    # print(means)

    for i in range(len(data)):
        for j in range(len(data)):
            C[i][j] = (1/len(data[i])) * np.sum(np.matmul((data[i] - np.mean(data[i])), (data[j] - np.mean(data[j])).transpose()))

    # print(C)
    # print(np.cov(data))

    return C


def SVD(A):

    # Computing U matrix
    AAT = np.matmul(A, A.transpose())

    eig_vals, eig_vecs = np.linalg.eig(AAT)
    # print(len(eig_vals), len(eig_vecs))
    eigs = {eig_vals[i]: eig_vecs[:,i] for i in range(len(eig_vals))}
    eigs = {eig_val: eig_vec for eig_val, eig_vec in sorted(eigs.items(), reverse=True)}

    U = np.array([eig_vec for eig_vec in eigs.values()])
    # idx1 = np.flip(np.argsort(eig_vals))
    # eig_vals = eig_vals[idx1]
    # U = eig_vecs[:, idx1]
    # print(U)
    # print(U)

    # Computing Sigma
    Sigma = np.zeros([len(eig_vals), len(eig_vals)+1])
    for i, eig_val in enumerate(eigs):
        Sigma[i][i] = np.sqrt(eig_val)
    # print(Sigma)

    # Computing V matrix
    ATA = np.matmul(A.transpose(), A)

    eig_vals, eig_vecs = np.linalg.eig(ATA)
    # print(len(eig_vals), len(eig_vecs))
    eigs = {eig_vals[i]: eig_vecs[:,i] for i in range(len(eig_vals))}
    eigs = {eig_val: eig_vec for eig_val, eig_vec in sorted(eigs.items(), reverse=True)}

    V = np.array([eig_vec for eig_vec in eigs.values()])
    # idx1 = np.flip(np.argsort(eig_vals))
    # eig_vals = eig_vals[idx1]
    # V = eig_vecs[:, idx1]
    # print(V)

    # print(U.shape, Sigma.shape, V.shape)
    # print((V[-1]/V[-1, -1]).reshape((3,3)))

    return U, Sigma, V


def linear_least_squares(data):

    # Equation of line: y = mx + x
    # print(data[1])
    X = np.array([[x, 1] for x in data[0]])
    Y = np.array([y for y in data[1]])
    a = np.empty([2,1])

    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y)
    # print(a)

    return X, Y, a


def total_least_squares(data):

    # Equation of line: ax + by + c = 0
    # print(data)
    X = np.array([[data[0][i], data[1][i]] for i in range(len(data[0]))])
    # a = np.empty([3,1])
    # print(X)

    U, Sigma, V = SVD(np.matmul(X.transpose(), X))
    u, sigma, v = np.linalg.svd(X)
    print(V)
    print(v)
    a = (V[:, -1]).reshape((2,1))
    print(a)

    # y = a[0]*data[0]+a[1]*data[1]
    # x = a[0]*data[0]
    y = (1 - 1*data[0])/1

    fig = plt.figure()
    plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
    plt.plot(data[0], y, 'red')
    plt.show()


def main():

    data = read_data("./Dataset/ENPM673_hw1_linear_regression_dataset.csv", ['age', 'charges']) # Get data from the CSV file

    cov_mat = compute_covariance_matrix(data) # Manually compute the Covariance Matric of the data

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    data[0] = (data[0] - data[0].min())/(data[0].max() - data[0].min())
    data[1] = (data[1] - data[1].min())/(data[1].max() - data[1].min())

    total_least_squares(data)

    # X, Y, a = linear_least_squares(data)

    # p = a[0]*data[0] + a[1]

    # origin = [0, 0]

    # fig = plt.figure()
    # plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
    # plt.plot(data[0], p, 'red')
    # plt.quiver(*origin, *eig_vecs[:,0], color=['r'], scale=21)
    # plt.quiver(*origin, *eig_vecs[1], color=['b'], scale=21)
    # plt.show()


if __name__ == '__main__':
    main()