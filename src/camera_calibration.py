# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-24 15:00:40
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-24 21:39:58

import os
from glob import glob

import cv2
import numpy as np


def calibrate_and_save_parameters(config, images_path_list, output_dir):
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(config['ARUCO_DICT'])
    board = cv2.aruco.CharucoBoard((config['SQUARES_VERTICALLY'],
                                    config['SQUARES_HORIZONTALLY']),
                                    config['SQUARE_LENGTH'],
                                    config['MARKER_LENGTH'],
                                    dictionary)
    params = cv2.aruco.DetectorParameters()

    all_charuco_corners = []
    all_charuco_ids = []

    h, w = cv2.imread(images_path_list[0]).shape[:2]
    for imagepath in images_path_list:
        image = cv2.imread(imagepath)
        image_copy = image.copy()

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

        # If at least one marker is detected
        if len(marker_corners) > 7:
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

            # cv2.imshow('drawDetectedMarkers', image_copy)
            # cv2.waitKey(0)

        # Calibrate camera
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners,
                                                                                            all_charuco_ids,
                                                                                            board,
                                                                                            (h, w),
                                                                                            None,
                                                                                            None)
    # Save calibration data
    np.save(os.path.join(output_dir, 'camera_matrix.npy'), camera_matrix)
    np.save(os.path.join(output_dir, 'dist_coeffs.npy'), dist_coeffs)

    # Iterate through displaying all the images
    for i, imagepath in enumerate(images_path_list):
        image = cv2.imread(imagepath)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.waitKey(0)
        if i > 10:
            break

    cv2.destroyAllWindows()

def main():
    # ------------------------------
    config = {}
    config['ARUCO_DICT'] = cv2.aruco.DICT_4X4_50
    config['SQUARES_VERTICALLY'] = 6
    config['SQUARES_HORIZONTALLY'] = 4
    config['SQUARE_LENGTH'] = 30 / 1000.0
    config['MARKER_LENGTH'] = 15 / 1000.0
    config['LENGTH_PX'] = 1123   # total length of the page in pixels
    config['MARGIN_PX'] = 20     # size of the margin in pixels

    images_dir = '/home/lcondados/workspace/arucodiscoveries/assets/frames'
    output_dir = '/home/lcondados/workspace/arucodiscoveries/assets'
    # ------------------------------

    images_path_list = glob(f'{images_dir}/*.jpg')
    images_path_list += glob(f'{images_dir}/*.png')

    assert len(images_path_list) != 0, 'Fail to find images in images_dir: {}'.format(images_dir)

    calibrate_and_save_parameters(config, images_path_list, output_dir)


if __name__ == "__main__":
    main()
