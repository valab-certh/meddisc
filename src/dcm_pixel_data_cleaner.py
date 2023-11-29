import os, logging

## Warning supression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

## Warning supression
tf.get_logger().setLevel(logging.ERROR)


import numpy as np
import cv2
import math
from time import time
import keras_ocr
import pandas as pd
import tensorflow as tf
import pydicom
from PIL import Image

import rw



def ndarray_size(arr: np.ndarray) -> int:
    return arr.itemsize*arr.size

def basic_preprocessing(img, downscale, toint8 = True, multichannel = True) -> np.ndarray:
    '''
        Description:
            Main preprocessing. It is imperative that the image is converted to (1) uint8 and in (2) RGB in order for keras_ocr's detector to properly function.

        Args:
            downscale. Bool.

        Returns:
            out_image. Its shape is (H, W) if `multichannel` is set to `False`, otherwise its shape is (H, W, 3).
    '''

    if downscale:
        ## Downscale
        downscale_dimensionality = 1024
        new_shape = (min([downscale_dimensionality, img.shape[0]]), min([downscale_dimensionality, img.shape[1]]))
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        print('Detection input downscaled to (%d, %d)'%(new_shape[0], new_shape[1]))

    if toint8:
        img = (255.0 * (img / np.max(img))).astype(np.uint8)

    if (multichannel) and (len(img.shape) == 2):
        img = np.stack(3*[img], axis = -1)

    return img

def text_remover(img, bboxes: np.ndarray, initial_array_shape, downscaled_array_shape):
    '''
        Args:
            bboxes. Shape (n_bboxes, 4, 2), where 4 is the number of vertices for each box and 2 are the plane coordinates. The vertices inside the bboxes array should be sorted in a way that corresponds to a geometrically counter-clockwise order. For example given a non-rotated (0 degree) bounding box with index 0, the following rule applies
                bboxes[0, 0, :] -> upper left vertex
                bboxes[0, 1, :] -> lower left vertex
                bboxes[0, 2, :] -> lower right vertex
                bboxes[0, 3, :] -> upper right vertex
    '''

    reducted_region_color = np.mean(img).astype(np.uint16)

    multiplicative_mask = np.ones(downscaled_array_shape, dtype = np.uint8)
    additive_mask = np.zeros(initial_array_shape, dtype = np.uint8)

    ## Concecutive embeddings of bounding boxes
    for bbox in bboxes:

        x0, y0 = bbox[0, 0:(1+1)]
        x1, y1 = bbox[1, 0:(1+1)]
        x2, y2 = bbox[2, 0:(1+1)]
        x3, y3 = bbox[3, 0:(1+1)]

        rectangle = np.array\
        (
            [
                [
                    [x0, y0],
                    [x1, y1],
                    [x2, y2],
                    [x3, y3]
                ]
            ],
            dtype = np.int32 ## Must remain this way. Otherwise, cv2.fillPoly will throw an error.
        )

        ## Filled rectangle
        cv2.fillPoly(multiplicative_mask, rectangle, 0)

    ## When multiplied with image, bounding box pixels will be replaced with 0
    multiplicative_mask = cv2.resize(multiplicative_mask, (initial_array_shape[1], initial_array_shape[0]), interpolation = cv2.INTER_NEAREST)

    ## When added after multiplication, bounding box pixels will be replaced with 255
    additive_mask = reducted_region_color * (multiplicative_mask == 0)

    img_ = img.copy()
    img_ = (img_ * multiplicative_mask + additive_mask)

    return img_

def keras_ocr_dicom_image_text_remover(dcm):

    def prep_det_keras_ocr(img):

        img_prep = basic_preprocessing(img = img, downscale = True)
        bboxes = det_keras_ocr(img_prep)

        return img_prep, bboxes

    def det_keras_ocr(img):

        pipeline = keras_ocr.detection.Detector()

        ## Returns a ndarray with shape (n_bboxes, 4, 2) where 4 is the number of points for each box, 2 are the plane coordinates.
        bboxes = pipeline.detect([img])[0]

        return bboxes

    t0 = time()

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    dcm.decompress()
    raw_img_uint16_grayscale = dcm.pixel_array

    ## Secondary information about the DICOM file
    print('Input DICOM file information')
    print('Input image shape: ', raw_img_uint16_grayscale.shape)

    t1 = time()

    raw_img_uint8_grayscale, bboxes = prep_det_keras_ocr(img = raw_img_uint16_grayscale)

    removal_period = time() - t1

    initial_array_shape = raw_img_uint16_grayscale.shape
    downscaled_array_shape = raw_img_uint8_grayscale.shape[:-1]

    if np.size(bboxes) != 0:

        cleaned_img = text_remover\
        (
            img = raw_img_uint16_grayscale,
            bboxes = bboxes,
            initial_array_shape = initial_array_shape,
            downscaled_array_shape = downscaled_array_shape
        )

        ## Update the DICOM image data with the modified image
        dcm.PixelData = cleaned_img.tobytes()

    else:

        print('Image state: No text detected')

    total_period = time() - t0

    return dcm, removal_period, total_period