import cv2
import numpy as np

def main():
    dest_imagepath  = r'C:\Users\lcondados\Documents\Novelis\Data\Marcacao_cones\PINDA\Camera8\Camera 8 - frame at 4m19s.jpg'
    board_imagepath = r'C:\Users\lcondados\Documents\Novelis\workspace\arucodiscoveries\src\ChArUco_Marker_4x4_50_larger.png'

    dest_image  = cv2.imread(dest_imagepath)
    board_image = cv2.imread(board_imagepath)

    board_h = board_image.shape[0]
    board_w = board_image.shape[1]

    imgH, imgW = dest_image.shape[:2]

    # pts_src = np.float32([(0,0), (0, board_w), (board_w, board_h), (0, board_h)])
    # pts_src = np.float32([(0,0), (board_w, 0), (board_w, board_h), (0, board_h)])
    pts_src = np.float32([(0,0), (0, board_w), (board_h, board_w), (board_h, 0)])
    pts_dst = np.float32([(72,484), (111,459), (135,473), (98,496)])

    # compute the homography matrix between the image and the video frame
    H, _ = cv2.findHomography(pts_src, pts_dst)

    #warp the  image to video frame based on the homography
    warped  = cv2.warpPerspective(board_image, H, (dest_image.shape[1], dest_image.shape[0]))

    #Create a mask representing region to
    #copy from the warped image into the video frame.
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, pts_dst.astype("int32"), (255, 255, 255),cv2.LINE_AA)

    # give the source image a black border
    # surrounding it when applied to the source image,
    #you can apply a dilation operation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)

    # Copy the mask with the three channel version by stacking it depth-wise,
    # This will allow copying the warped source image into the input image
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)

    # Copy the masked warped image into the video frame by
    # (1) multiplying the warped image and masked together,
    # (2) multiplying the Video frame with the mask
    # (3) adding the resulting images
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(dest_image.astype(float), 1.0 - maskScaled)
    #imgout = video frame multipled with mask
    #        + warped image multipled with mask
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    cv2.imshow("output", output)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()