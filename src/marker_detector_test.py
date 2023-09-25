# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-06-11 21:01:43
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-09-23 02:02:49

import cv2
import numpy as np

def gen_ArUco_marker(dictionary, id, size):
    markerImage = cv2.aruco.generateImageMarker(dictionary, id, size, None, 1)
    return markerImage

def main():
    # load predefined dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # size = 200
    # for i in range(50):
    #     markerImage = gen_ArUco_marker(dictionary, i, size)
    #     filename = Path(f'./markers/{i}.jpg')
    #     cv2.imwrite(filename.as_posix(), markerImage)

    # cv2.imshow('Marker', markerImage)
    # cv2.waitKey(0)

    video = cv2.VideoCapture('/dev/video0')

    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        cv2.imshow('webcam', frame)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # (markerCorners, markerIds, rejectedCandidates) = detector.detectMarkers(gray)
        (markerCorners, markerIds, rejectedCandidates) = detector.detectMarkers(frame)

        if markerCorners != ():
            markerIds = markerIds.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(markerCorners, markerIds):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(frame,
                            str(markerID),
                            (topLeft[0], topLeft[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                # print("[INFO] ArUco marker ID: {}".format(markerID))
                # show the output image
                cv2.imshow("webcam", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

if __name__ == '__main__':
    main()