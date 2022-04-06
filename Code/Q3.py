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
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    U = eig_vecs
    # print(U)

    # Computing Sigma
    eig_vals = eig_vals[np.where(eig_vals > 0)]
    eig_vals = np.sqrt(eig_vals)
    Sigma = np.zeros(A.shape)
    np.fill_diagonal(Sigma, eig_vals)
    # print(Sigma)

    # Computing V matrix
    ATA = np.matmul(A.transpose(), A)

    eig_vals, eig_vecs = np.linalg.eig(ATA)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    V = eig_vecs
    # print(V)

    # print(U.shape, Sigma.shape, V.shape)
    # print((V[-1]/V[-1, -1]).reshape((3,3)))

    return U, Sigma, V


def linear_least_squares(data, Visualize=False):

    if(Visualize):
        print("Applying Linear Least Squares...")

    # Equation of line: y = mx + c
    # print(data[1])
    X = np.array([[x, 1] for x in data[0]])
    Y = np.array([y for y in data[1]])
    a = np.empty([2,1])

    a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.transpose(), X)), X.transpose()), Y)
    # print(a)

    y = a[0]*data[0] + a[1]

    # origin = [0, 0]

    if(Visualize):
        fig = plt.figure()
        plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
        plt.plot(data[0], y, 'red')
        plt.show()

    return a, y


def total_least_squares(data, Visualize=False):

    print("Applying Total Least Squares...")

    # Equation of line: ax + by - d = 0
    # Matrix U = [x_i - x_mean, y_i - y_mean]
    # Matrix N = [a b]
    # print(data)
    x_mean = np.mean(data[0])
    y_mean = np.mean(data[1])

    X = np.array([[data[0][i] - x_mean, data[1][i] - y_mean] for i in range(len(data[0]))])

    U, Sigma, V = SVD(np.matmul(X.transpose(), X))

    a = (V[:, -1]).reshape((2,1))
    d = a[0]*x_mean + a[1]*y_mean
    # print(a, d)
    y = (d - a[0]*data[0])/a[1]

    if(Visualize):
        fig = plt.figure()
        plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
        plt.plot(data[0], y, 'red')
        plt.show()

    return a, y


def RANSAC(data, Visualize=False):

    print("Applying RANSAC...")

    p = 0.95 # Probability of success
    e = 0.6 # Outlier Ratio
    N = int(np.log(1-p)/np.log(1-((1-e)**2)))
    threshold = 0.01
    max_inliers = 0
    best_a = None
    best_Y = None
    # print(N)

    idxs = np.arange(len(data[0])).tolist()
    for i in range(N):
        points = list()
        idx = np.random.choice(idxs, 2, replace=False)
        for p in idx:
            points.append([data[0][p], data[1][p]])

        points = np.array(points).transpose()

        a, c = linear_least_squares(points)

        Y = a[0]*data[0] + a[1]

        SSD = list()
        SSD = (Y-data[1])**2
        for i in range(len(SSD)):
            if(SSD[i] <= threshold):
                SSD[i] = 1
            else:
                SSD[i] = 0

        # print(SSD)
        inliers = np.sum(SSD)
        if(inliers > max_inliers):
            max_inliers = inliers
            best_a = a
            best_Y = Y
            # print(max_inliers)
    
    print("Max Inliers: %d"%max_inliers)

    if(Visualize):
        fig = plt.figure()
        plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
        plt.plot(data[0], best_Y, 'red')
        plt.show()

    return best_a, best_Y


def main():

    data = read_data("./Dataset/ENPM673_hw1_linear_regression_dataset.csv", ['age', 'charges']) # Get data from the CSV file

    cov_mat = compute_covariance_matrix(data) # Manually compute the Covariance Matric of the data

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    data[0] = (data[0] - data[0].min())/(data[0].max() - data[0].min())
    data[1] = (data[1] - data[1].min())/(data[1].max() - data[1].min())

    print("Applying Linear Least Squares...")
    a, LLS_Y = linear_least_squares(data, False)
    
    a, TLS_Y = total_least_squares(data, False)

    a, RNS_Y = RANSAC(data, False)

    origin = [np.mean(data[0]), np.mean(data[1])]

    fig = plt.figure()
    plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
    plt.plot(data[0], LLS_Y, 'red')
    plt.plot(data[0], TLS_Y, 'blue')
    plt.plot(data[0], RNS_Y, 'orange')
    plt.quiver(*origin, *eig_vecs[:,0], color=['gray'], scale=10)
    plt.quiver(*origin, *eig_vecs[1], color=['gray'], scale=10)
    plt.legend(["Data", "Linear Least Squares", "Total Least Squares", "RANSAC", "Eigen Vectors of Covariance Mat"])
    plt.show()


if __name__ == '__main__':
    main()