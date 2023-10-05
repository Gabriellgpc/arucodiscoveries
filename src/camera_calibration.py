# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-24 15:00:40
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-24 21:39:58

import os
from glob import glob

import cv2
import numpy as np
import time
from tqdm import tqdm

from uuid import uuid4

from aruco_detectors import CharucoDetector

def camera_calibrate(config, images_path_list):
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
    for imagepath in tqdm(images_path_list):

        image = cv2.imread(imagepath)
        image_copy = image.copy()

        # print(imagepath)
        # print(image.shape)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image,
                                                                dictionary,
                                                                parameters=params)

        # If at least one marker is detected
        if len(marker_corners) > 0:
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
    # Iterate through displaying all the images
    for i, imagepath in enumerate(images_path_list):
        image = cv2.imread(imagepath)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.waitKey(0)
        if i > 10:
            break
    cv2.destroyAllWindows()

    return camera_matrix, dist_coeffs

def camera_calibration_from_images_dir(config, images_dir):
    images_path_list = glob(f'{images_dir}/*.jpg')
    images_path_list += glob(f'{images_dir}/*.png')

    assert len(images_path_list) != 0, 'Fail to find images in images_dir: {}'.format(images_dir)

    camera_matrix, dist_coeffs = camera_calibrate(config, images_path_list)
    return camera_matrix, dist_coeffs

def camera_calibration_from_stream(config, video_stream, max_frames=25):
    """
        It will read the video stream and save all frames where
        the program can detect the expected charuco board.
    """

    # Create a new folder to store the frames containing charuco board
    images_dir = os.path.join('assets/frames_automatic_saved-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(images_dir)

    # open the video stream
    video = cv2.VideoCapture(video_stream)

    # create charuco board detector
    board_detector = CharucoDetector(config)

    print('[INFO] Taking photos of the charuco board...')
    frames_saved = 0
    prog_bar = tqdm(total=max_frames)
    prev_timestamp = time.time()
    while True:

        if frames_saved >= max_frames:
            print(frames_saved)
            break

        ret, frame = video.read()
        if ret == False: break
        viz_image = frame.copy()

        # pass to charuco board detector
        charuco_retval, charuco_corners, charuco_ids = board_detector.detect(frame)
        if charuco_retval:

            for idx, charuco_corner in enumerate(charuco_corners):
                xc = int(charuco_corner[0][0])
                yc = int(charuco_corner[0][1])

                color = (0, 255, 0) # BGR

                cv2.circle(viz_image, (xc, yc), 5, color, -1)

                corner_id = charuco_ids[idx]
                cv2.putText(viz_image, '{}'.format(corner_id), (xc-10, yc-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # print(time.time() - prev_timestamp)
                if (time.time() - prev_timestamp) >= 5:
                    filename = '{}.jpg'.format(uuid4().hex)
                    imagepath = os.path.join(images_dir, filename)
                    cv2.imwrite(imagepath, frame)

                    prog_bar.update(1)
                    frames_saved += 1
                    prev_timestamp = time.time()

        cv2.imshow('input', viz_image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break


    print('[INFO] Computing camera parameters...')
    if frames_saved == 0:
        return None, None
    images_path_list = glob(f'{images_dir}/*.jpg')
    camera_matrix, dist_coeffs = camera_calibrate(config, images_path_list)
    return camera_matrix, dist_coeffs

def main():
    # ------------------------------
    config = {}
    config['ARUCO_DICT'] = cv2.aruco.DICT_4X4_50
    config['SQUARES_VERTICALLY'] = 6
    config['SQUARES_HORIZONTALLY'] = 4
    config['SQUARE_LENGTH'] = 317 / 1000.0
    config['MARKER_LENGTH'] = 254 / 1000.0
    config['LENGTH_PX'] = 1123   # total length of the page in pixels
    config['MARGIN_PX'] = 20     # size of the margin in pixels

    video_stream = 0 #webcam video stream
    images_dir = 'assets/frames'
    output_dir = 'assets/webcam_parameters'
    # ------------------------------

    os.makedirs(output_dir, exist_ok=True)

    # camera_matrix, dist_coeffs = camera_calibration_from_images_dir(config, images_dir)
    camera_matrix, dist_coeffs = camera_calibration_from_stream(config, video_stream)

    # Save calibration data
    np.save(os.path.join(output_dir, 'camera_matrix.npy'), camera_matrix)
    np.save(os.path.join(output_dir, 'dist_coeffs.npy'), dist_coeffs)

if __name__ == "__main__":
    main()
