#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Homework 1: Question 2

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


def get_data_points():

    video = cv2.VideoCapture("./Videos/ball_video1.mp4") 

    currentframe = 0
    frames = list() # Frames
    x_coords = list() # X Coordinates of center of the ball in Image plane
    y_coords = list() # Y Coordinates of center of the ball in Image plane
    ret = True

    X = np.empty([0,3])
    Y = np.empty([0,1])
    a = np.empty([3,1])

    while(ret): 

        ret, frame = video.read()
        if(not ret):
            break
        frame = cv2.resize(frame, dsize=None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
         
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
         
        lower_mask = cv2.inRange(hsv_frame, lower1, upper1)
        upper_mask = cv2.inRange(hsv_frame, lower2, upper2)
         
        full_mask = lower_mask + upper_mask;

        currentframe += 1
        frames.append(currentframe)

        test = np.array(np.where(full_mask == 255))

        # print(full_mask.shape)
        # print(test.shape)
        # print(test[0].argmax(), test[1].argmax())
        # print(test[0].argmin(), test[1].argmin())
        # print(test[0][test[0].argmax()], test[1][test[1].argmax()])
        # print(test[0][test[0].argmin()], test[1][test[1].argmin()])

        x = test[1][test[1].argmin()] + (test[1][test[1].argmax()] - test[1][test[1].argmin()])/2 # X/columns axis in Image plane, Y/rows in a matrix
        y = test[0][test[0].argmin()] + (test[0][test[0].argmax()] - test[0][test[0].argmin()])/2 # Y/rows axis in Image plane, X/columns in a matrix

        # print(x, y)

        full_mask = cv2.circle(frame, (int(x), int(y)), 2, (255,0,0), -1)
        
        x_coords.append(x)
        y_coords.append(y)

        # print(x**2, x, y)
        X = np.insert(X, len(X), np.array([x**2, x, 1]), axis=0)
        Y = np.insert(Y, len(Y), y, axis=0)
        # print(X.shape, Y.shape)

        cv2.imshow("Frame", frame)
        cv2.imshow("Gray Frame", full_mask)
        cv2.imshow("HSV Frame", hsv_frame)
        cv2.waitKey(1)

    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y)
    print(a)

    x_coords_sq = [x**2 for x in x_coords]
    p = a[0]*x_coords_sq + a[1]*x_coords + a[2]

    fig = plt.figure()
    plt.scatter(x_coords, y_coords, linewidths=0.2)
    plt.plot(x_coords, p)
    plt.show()

    video.release() 
    cv2.destroyAllWindows()


def main():

    get_data_points()


if __name__ == '__main__':
    main()