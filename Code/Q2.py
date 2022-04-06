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


def get_data_points(VideoPath):

    video = cv2.VideoCapture(VideoPath) 

    currentframe = 0
    frames = list() # Frames
    x_coords = list() # X Coordinates of center of the ball in Image plane
    y_coords = list() # Y Coordinates of center of the ball in Image plane
    ret = True

    # System of Linear Equations
    X = np.empty([0,3])
    Y = np.empty([0,1])
    a = np.empty([3,1])

    while(ret): 

        ret, frame = video.read()
        if(not ret):
            break
        # frame = cv2.resize(frame, dsize=None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
         
        # Upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
         
        lower_mask = cv2.inRange(hsv_frame, lower1, upper1)
        upper_mask = cv2.inRange(hsv_frame, lower2, upper2)
         
        full_mask = lower_mask + upper_mask;

        currentframe += 1
        frames.append(currentframe)

        PoI = np.array(np.where(full_mask == 255)) # Points of interest

        x = PoI[1][PoI[1].argmin()] + (PoI[1][PoI[1].argmax()] - PoI[1][PoI[1].argmin()])/2 # X/columns axis in Image plane, Y/rows in a matrix
        y = PoI[0][PoI[0].argmin()] + (PoI[0][PoI[0].argmax()] - PoI[0][PoI[0].argmin()])/2 # Y/rows axis in Image plane, X/columns in a matrix

        full_mask = cv2.circle(frame, (int(x), int(y)), 2, (255,0,0), -1)

        y = abs(y - frame.shape[0]) # Transforming points to cartesian coordinates

        x_coords.append(x)
        y_coords.append(y)

        X = np.insert(X, len(X), np.array([x**2, x, 1]), axis=0)
        Y = np.insert(Y, len(Y), y, axis=0)

        # cv2.imshow("Frame", frame)
        # cv2.imshow("Gray Frame", full_mask)
        # cv2.imshow("HSV Frame", hsv_frame)
        # cv2.waitKey(1)

    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y) # Solving the system of linear equations for a

    x_coords_sq = [x**2 for x in x_coords]
    p = a[0]*x_coords_sq + a[1]*x_coords + a[2] # Generating points of the parabola

    fig = plt.figure()
    plt.scatter(x_coords, y_coords, linewidths=0.2)
    plt.plot(x_coords, p, 'red')
    plt.show()

    video.release() 
    cv2.destroyAllWindows()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="./Videos/ball_video1.mp4", help='Path to the video file')
    
    Args = Parser.parse_args()
    VideoPath = Args.VideoPath

    get_data_points(VideoPath)


if __name__ == '__main__':
    main()