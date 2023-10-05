# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-24 16:09:25
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-24 23:28:04

import cv2
import numpy as np

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

class ArucoDetector(object):
    def __init__(self, config_dict):
        pass
    def detect(self, image):
        pass

class CharucoDetector(object):
    def __init__(self, config):
        self.config = config

        # Define the aruco dictionary and charuco board
        self.dictionary = cv2.aruco.getPredefinedDictionary(config['ARUCO_DICT'])
        self.board = cv2.aruco.CharucoBoard((config['SQUARES_VERTICALLY'],
                                             config['SQUARES_HORIZONTALLY']),
                                             config['SQUARE_LENGTH'],
                                             config['MARKER_LENGTH'],
                                             self.dictionary)
        self.params = cv2.aruco.DetectorParameters()

    def detect(self, image):
        # Detect markers in the undistorted image
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image,
                                                                self.dictionary,
                                                                parameters=self.params)

        charuco_retval, charuco_corners, charuco_ids = 0, [], []
        if len(marker_corners) > 0:
            # Interpolate CharUco corners
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners,
                                                                                               marker_ids,
                                                                                               image,
                                                                                               self.board)
            # print('marker_ids', marker_ids)
            # print('marker_ids: {} | charuco_ids: {}'.format(marker_ids, charuco_ids))
            # print('marker_corners: {}'.format(np.array(marker_corners).shape))
            # print('charuco_corners: {}'.format(np.array(charuco_corners).shape))
            # if charuco_retval:
            #     for charuco_corner in charuco_corners:
            #         xc = int(charuco_corner[0][0])
            #         yc = int(charuco_corner[0][1])
            #         cv2.circle(image, (xc, yc), 10, (0, 255, 0), -1)
            # print('marker_corners: {} | charuco_corners: {}'.format(marker_corners, charuco_corners))
        # return charuco_retval, marker_corners, marker_ids, charuco_corners, charuco_ids
        # return charuco_retval, marker_corners, marker_ids
        return charuco_retval, charuco_corners, charuco_ids

    def estimate_pose(self,
                      image,
                      marker_corners,
                      marker_ids,
                      camera_matrix,
                      dist_coeffs,
                      draw_on_image=False):

        if draw_on_image:
            cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

        for i, marker_id in enumerate(marker_ids):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i],
                                                                self.config['MARKER_LENGTH'],
                                                                camera_matrix,
                                                                dist_coeffs)

            for corner in marker_corners[i].reshape(-1,2):
                cv2.circle(image, tuple(np.int32(corner)), 2, (0,255,255), -1)

            cv2.drawFrameAxes(image,
                              camera_matrix,
                              dist_coeffs,
                              rvec,
                              tvec,
                              0.02)
            square_size = self.config['SQUARE_LENGTH']
            axisBoxes = np.float32([
                                    [-square_size, square_size, 0],
                                    [square_size, square_size, 0],
                                    [square_size,-square_size, 0],
                                    [-square_size,-square_size, 0],
                                    [-square_size, square_size, 2*square_size],
                                    [square_size, square_size, 2*square_size],
                                    [square_size,-square_size, 2*square_size],
                                    [-square_size,-square_size, 2*square_size],
                                   ])

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axisBoxes, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype('int32').reshape(-1, 2)
            drawBoxes(image, marker_corners[i], imgpts)