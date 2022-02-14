#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Homework 1: Question 4

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


def SVD(A):

    # Computing U matrix
    AAT = np.matmul(A, A.transpose())

    eig_vals, eig_vecs = np.linalg.eig(AAT)
    eigs = {eig_vals[i]: eig_vecs[:,i] for i in range(len(eig_vals))}
    eigs = {eig_val: eig_vec for eig_val, eig_vec in sorted(eigs.items(), reverse=True)}

    U = np.array([eig_vec for eig_vec in eigs.values()])
    print(U)

    idx1 = np.flip(np.argsort(eig_vals))
    eig_vals = eig_vals[idx1]
    U = eig_vecs[:, idx1]
    print(U)

    # Computing Sigma
    Sigma = np.zeros([len(eig_vals), len(eig_vals)+1])
    for i, eig_val in enumerate(eigs):
        Sigma[i][i] = np.sqrt(eig_val)
    # print(Sigma)

    # Computing V matrix
    ATA = np.matmul(A.transpose(), A)

    eig_vals, eig_vecs = np.linalg.eig(ATA)
    eigs = {eig_vals[i]: eig_vecs[:,i] for i in range(len(eig_vals))}
    eigs = {eig_val: eig_vec for eig_val, eig_vec in sorted(eigs.items(), reverse=True)}

    V = np.array([eig_vec for eig_vec in eigs.values()])
    idx1 = np.flip(np.argsort(eig_vals))
    eig_vals = eig_vals[idx1]
    V = eig_vecs[:, idx1]
    print(V)

    print(U.shape, Sigma.shape, V.shape)
    # print((V[-1]/V[-1, -1]).reshape((3,3)))

    return U, Sigma, V


def HomographyMatrix(V):
    return (V[-1]/V[-1, -1]).reshape((3,3))


def main():

    x = np.array([5, 150, 150, 5]).transpose()
    y = np.array([5, 5, 150, 150]).transpose()
    xp = np.array([100, 200, 220, 100]).transpose()
    yp = np.array([100, 80, 80, 200]).transpose()

    A = np.array([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                  [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                  [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                  [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                  [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                  [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
                  [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                  [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])
    
    U, Sigma, V = SVD(A)
    print(V)

    u, s, v = np.linalg.svd(A)
    print(v)

    H = HomographyMatrix(V)
    print("Homography Matrix:\n", H)

if __name__ == '__main__':
    main()