import cv2
import numpy as np
import bar_decoder
import time


def start(image):

    # load the image and convert it to grayscale
    #start = time.time()
    try:
        #cv2.imshow("90", image_90)
        box, rect = bar_detector(image)
        if box.min() > 20:
            answer = send_crop(image, box, rect)
            #cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
            #cv2.imshow("image", image)
            if answer:
                cv2.putText(image, answer[0], (min(box[:, 0]), min(box[:, 1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
                return answer, image
        image_90 = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        box_90, rect_90 = bar_detector(image_90)
        if box_90.min() > 20:
            answer = send_crop(image_90, box_90, rect_90)
            if answer:
                height, width = image.shape[:2]
                box_90 = box_90[:, ::-1]
                box_90[0][1] = height - box_90[0][1]
                box_90[1][1] = height - box_90[1][1]
                box_90[2][1] = height - box_90[2][1]
                box_90[3][1] = height - box_90[3][1]
                cv2.putText(image, answer[0], (min(box_90[:, 0]), min(box_90[:, 1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.drawContours(image, [box_90], -1, (0, 255, 0), 3)
                return answer, image

    except:
        pass

    return "cannot encode", image


def bar_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    abs_gradX = np.absolute(gradX)
    gradX = np.uint8(abs_gradX)
    abs_gradY = np.absolute(gradY)
    gradY = np.uint8(abs_gradY)
    #cv2.imshow('gradX', gradX)
    #cv2.imshow('gradY', gradY)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    #cv2.imshow('gradient', gradient)
    # blur and threshold the image
    blurred = cv2.medianBlur(gradient, 19)
    #cv2.imshow('blurred', blurred)
    (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    #cv2.imshow('thresh',thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 9))
    erode = cv2.erode(thresh, kernel, iterations=1)
    #cv2.imshow('erode', erode)
    dilate = cv2.dilate(erode, kernel, iterations=2)
    closed = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('closing', closed)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box, rect


def cropping(image, box, rect):
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))
    height, width = warped.shape[:2]
    if width < height:
        warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    #cv2.imshow('warped', warped)
    return warped


def send_crop(image, box, rect):
    to_decode = cropping(image, box, rect)
    #cv2.imwrite("to_decode.jpg", to_decode)
    answer = bar_decoder.run(to_decode)
    return answer
