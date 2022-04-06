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

np.set_printoptions(precision=4, suppress=True)

def SVD(A):

    # Computing U matrix
    AAT = np.matmul(A, A.transpose())
    # print(AAT)

    eig_vals, eig_vecs = np.linalg.eig(AAT)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    U = eig_vecs
    print("Matrix U:\n", U)

    # Computing Sigma
    eig_vals = eig_vals[np.where(eig_vals > 0)]
    eig_vals = np.sqrt(eig_vals)
    Sigma = np.zeros(A.shape)
    np.fill_diagonal(Sigma, eig_vals)
    print("Matrix Sigma:\n", Sigma)

    # Computing V matrix
    ATA = np.matmul(A.transpose(), A)
    # print(ATA)

    eig_vals, eig_vecs = np.linalg.eig(ATA)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    V = eig_vecs
    print("Matrix V^T:\n", V.transpose())

    # print(U.shape, Sigma.shape, V.shape)
    # print((V[-1]/V[-1, -1]).reshape((3,3)))

    return U, Sigma, V.transpose()


def HomographyMatrix(V):
    # print((V[:,-1]).reshape((3,3)))
    return (V[:,-1]/V[:,-1,][-1]).reshape((3,3))


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
    
    print("Computing SVD...")
    U, Sigma, V = SVD(A)

    H = HomographyMatrix(V.transpose())
    print("Homography Matrix:\n", H)

if __name__ == '__main__':
    main()