# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-09-23 02:02:38
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-24 20:38:14

# Reference
# https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html

# The ArUco module can also be used to calibrate a camera.
# Camera calibration consists in obtaining the camera intrinsic parameters and distortion coefficients.
# This parameters remain fixed unless the camera optic is modified, thus camera calibration only need to be done once.
# Camera calibration is usually performed using the OpenCV calibrateCamera() function.
# This function requires some correspondences between environment points and their projection in the camera image from different viewpoints.
# In general, these correspondences are obtained from the corners of chessboard patterns.
# See calibrateCamera() function documentation or the OpenCV calibration tutorial for more detailed information.
# Using the ArUco module, calibration can be performed based on ArUco markers corners or ChArUco corners.
# Calibrating using ArUco is much more versatile than using traditional chessboard patterns, since it allows occlusions or partial views.
# As it can be stated, calibration can be done using both, marker corners or ChArUco corners.
# However, it is highly recommended using the ChArUco corners approach since the provided corners are much more accurate in comparison to the marker corners.
# Calibration using a standard Board should only be employed in those scenarios where the ChArUco boards cannot be employed because of any kind of restriction.


# To calibrate using a ChArUco board, it is necessary to detect the board from different viewpoints,
# in the same way that the standard calibration does with the traditional chessboard pattern.
# However, due to the benefits of using ChArUco, occlusions and partial views are allowed, and not all the corners need to be visible in all the viewpoints.

import cv2
import cv2.aruco as aruco

import common

def create_new_charuco_board(config_dict):
    """Sample:
        config = {}
            config['ARUCO_DICT'] = cv2.aruco.DICT_4X4_50
            config['SQUARES_VERTICALLY'] = 7
            config['SQUARES_HORIZONTALLY'] = 5
            config['SQUARE_LENGTH'] = 30 / 1000.0
            config['MARKER_LENGTH'] = 15 / 1000.0
            config['LENGTH_PX'] = 1123   # total length of the page in pixels
            config['MARGIN_PX'] = 20    # size of the margin in pixels
            config['SAVE_NAME'] = 'ChArUco_Marker.png'
    """

    dictionary = cv2.aruco.getPredefinedDictionary(config_dict['ARUCO_DICT'])
    board = cv2.aruco.CharucoBoard((config_dict['SQUARES_VERTICALLY'],
                                    config_dict['SQUARES_HORIZONTALLY']),
                                    config_dict['SQUARE_LENGTH'],
                                    config_dict['MARKER_LENGTH'],
                                    dictionary)
    size_ratio = config_dict['SQUARES_HORIZONTALLY'] / config_dict['SQUARES_VERTICALLY']
    image = cv2.aruco.CharucoBoard.generateImage(board,
                                                (config_dict['LENGTH_PX'],
                                                 int(config_dict['LENGTH_PX']*size_ratio)),
                                                 marginSize=config_dict['MARGIN_PX'])

    return board, image

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
    config['SAVE_NAME'] = 'ChArUco_Marker.png'
    # ------------------------------

    _, charuco_image = create_new_charuco_board(config)

    text = 'Pattern: {}x{} | Square Size: {:.1f} mm | Marker: {:.1f} mm | Dictionary: {}'
    text = text.format(config['SQUARES_HORIZONTALLY'],
                       config['SQUARES_VERTICALLY'],
                       config['SQUARE_LENGTH']*1000.0,
                       config['MARKER_LENGTH']*1000.0,
                       'DICT_4X4_50'
                       )
    h, w  = charuco_image.shape[:2]
    org = (config['MARGIN_PX']//2, h-config['MARGIN_PX']//2)

    cv2.putText(charuco_image,
                text=text,
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.30,
                color=(27,27,27),
                thickness=1)

    cv2.imwrite(config['SAVE_NAME'] , charuco_image)

    cv2.imshow("charuco_image", charuco_image)
    cv2.waitKey(0)

if __name__=='__main__':
    main()