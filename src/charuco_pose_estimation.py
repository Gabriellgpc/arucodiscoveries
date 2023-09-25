# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-24 16:09:25
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-24 23:28:04

# Axis X (red color) - first coordinate, axis Y (green color) - second coordinate, axis Z (blue color)

import numpy as np
import cv2

from glob import glob

def drawBoxes(image, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    image = cv2.drawContours(image, [imgpts[:4]], -1, (0,255,0), -1)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4,8)):
        image = cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0),2)
    # draw top layer in red color
    image = cv2.drawContours(image, [imgpts[4:]], -1, (0,0,255),2)
    return image

def detect_pose(config, image, camera_matrix, dist_coeffs):
    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(config['ARUCO_DICT'])
    board = cv2.aruco.CharucoBoard((config['SQUARES_VERTICALLY'],
                                    config['SQUARES_HORIZONTALLY']),
                                    config['SQUARE_LENGTH'],
                                    config['MARKER_LENGTH'],
                                    dictionary)
    params = cv2.aruco.DetectorParameters()

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image,
                                                            dictionary,
                                                            parameters=params)

    # If at least one marker is detected
    if len(marker_corners) > 0:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners,
                                                                                           marker_ids,
                                                                                           undistorted_image,
                                                                                           board)

        cv2.aruco.drawDetectedMarkers(undistorted_image, marker_corners, marker_ids)
        for i, marker_id in enumerate(marker_ids):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], 30/1000.0, camera_matrix, dist_coeffs)
            print('marker_corners[i]', marker_corners[i])
            print('marker_corners[i].shape', marker_corners[i].shape)
            for corner in marker_corners[i].reshape(-1,2):
                cv2.circle(undistorted_image, tuple(np.int32(corner)), 2, (0,255,255), -1)
            cv2.drawFrameAxes(undistorted_image,
                              camera_matrix,
                              dist_coeffs,
                              rvec,
                              tvec,
                              0.02)
            square_size = 15/1000.0
            axisBoxes = np.float32([
                                    [-square_size, square_size, 0],
                                    [square_size, square_size, 0],
                                    [square_size,-square_size, 0],
                                    [-square_size,-square_size, 0],
                                    [-square_size, square_size, 2*square_size],
                                    [square_size, square_size, 2*square_size],
                                    [square_size,-square_size, 2*square_size],
                                    [-square_size,-square_size, 2*square_size],
                                   ]
                                    )
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axisBoxes, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype('int32').reshape(-1, 2)
            drawBoxes(undistorted_image, marker_corners[i], imgpts)


    # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners,
                                                                    charuco_ids,
                                                                    board,
                                                                    camera_matrix,
                                                                    dist_coeffs,
                                                                    None,
                                                                    None)

            # If pose estimation is successful, draw the axis
            if retval:
                camera_distance = np.linalg.norm(tvec)
                text = '({:.2f}, {:.2f}, {:.2f}) [m] | Camera distance: {:.2f} m'.format(*tvec.flatten(), camera_distance)
                print(text)
                cv2.drawFrameAxes(undistorted_image,
                                  camera_matrix,
                                  dist_coeffs,
                                  rvec,
                                  tvec,
                                  length=0.1,
                                  thickness=5)

    return undistorted_image

def test_on_images_dir():
    # Load calibration data
    camera_matrix = np.load('/home/lcondados/workspace/arucodiscoveries/assets/camera_matrix.npy')
    dist_coeffs = np.load('/home/lcondados/workspace/arucodiscoveries/assets/dist_coeffs.npy')
    # images dir
    images_dir = '/home/lcondados/workspace/arucodiscoveries/assets/frames'

    config = {}
    config['ARUCO_DICT'] = cv2.aruco.DICT_4X4_50
    config['SQUARES_VERTICALLY'] = 6
    config['SQUARES_HORIZONTALLY'] = 4
    config['SQUARE_LENGTH'] = 30 / 1000.0
    config['MARKER_LENGTH'] = 15 / 1000.0

    # read all images path under images_dir
    images_path_list = glob(f'{images_dir}/*.jpg')
    images_path_list += glob(f'{images_dir}/*.png')

    assert len(images_path_list) != 0, 'Fail to find images in images_dir: {}'.format(images_dir)

    for imagepath in images_path_list:
        # Load an image
        image = cv2.imread(imagepath)

        # Detect pose and draw axis
        pose_image = detect_pose(config, image, camera_matrix, dist_coeffs)

        # Show the image
        cv2.imshow('Pose Image', pose_image)
        cv2.waitKey(0)

def test_on_video():
    # Load calibration data
    camera_matrix = np.load('/home/lcondados/workspace/arucodiscoveries/assets/camera_matrix.npy')
    dist_coeffs = np.load('/home/lcondados/workspace/arucodiscoveries/assets/dist_coeffs.npy')

    config = {}
    config['ARUCO_DICT'] = cv2.aruco.DICT_4X4_50
    config['SQUARES_VERTICALLY'] = 6
    config['SQUARES_HORIZONTALLY'] = 4
    config['SQUARE_LENGTH'] = 30 / 1000.0
    config['MARKER_LENGTH'] = 15 / 1000.0

    video = cv2.VideoCapture('/dev/video0')

    while True:
        ret, frame = video.read()
        if ret == False: break
        # frame = frame[:,::-1,:]

        undistorted_image = detect_pose(config, frame, camera_matrix, dist_coeffs)

        cv2.imshow('video', undistorted_image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

if __name__ == '__main__':
    # test_on_images_dir()
    test_on_video()