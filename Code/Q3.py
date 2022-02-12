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


def linear_least_squares(data):

    # Equation of line: y = mx + x
    # print(data[1])
    X = np.array([[x, 1] for x in data[0]])
    Y = np.array([y for y in data[1]])
    a = np.empty([2,1])

    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y)
    # print(a)

    return X, Y, a


def main():

    data = read_data("./Dataset/ENPM673_hw1_linear_regression_dataset.csv", ['age', 'charges']) # Get data from the CSV file

    cov_mat = compute_covariance_matrix(data) # Manually compute the Covariance Matric of the data

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    X, Y, a = linear_least_squares(data)

    # x_coords_sq = [x**2 for x in data[0]]
    p = a[0]*data[0] + a[1]

    origin = [0, 0]

    fig = plt.figure()
    plt.scatter(data[0], data[1], marker='.', linewidths=0.01)
    plt.plot(data[0], p)
    # plt.quiver(*origin, *eig_vecs[0], color=['r'], scale=21)
    # plt.quiver(*origin, *eig_vecs[1], color=['b'], scale=21)
    plt.show()


if __name__ == '__main__':
    main()