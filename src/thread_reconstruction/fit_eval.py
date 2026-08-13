import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.interpolate as interp
import cv2
import os
import torch

from src.thread_reconstruction.segmenter import UNetSegmenter
from src.thread_reconstruction.keypt_selection import keypt_selection
from src.thread_reconstruction.keypt_ordering import keypt_ordering
from src.thread_reconstruction.optim import optim
from src.thread_reconstruction.utils import *

"""
img1: RGB left camera image (np array)
img2: RGB right camera image (np array)
calib: filename for camera calibration file (string)
segmenter: segmentation object (see segmenter.py)
"""
def fit_eval(img1, img2, calib, segmenter):
    # Read in camera matrix
    cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    ImageSize = cv_file.getNode("ImageSize").mat()
    img_size = (int(ImageSize[0][1]), int(ImageSize[0][0]))
    new_size = (640, 480)

    # Rectify image and store necessary matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
    cam2img = P1[:,:-1]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, new_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, new_size, cv2.CV_16SC2)
    img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    # Segment stereo images
    mask1 = segmenter.segmentation(img1)
    mask2 = segmenter.segmentation(img2)
    stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
    img1 = np.where(stack_mask1>0, img1, 0)
    stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
    img2 = np.where(stack_mask2>0, img2, 0)
    
    # Convert from btyes to float
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Perform reconstruction
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, Q)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, cluster_map, keypoints, grow_paths, adjacents)
    spline, reliability = optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, cam2img, P1, P2)
    return spline

if __name__ == "__main__":
    USE_SAM = False

    # inp_folder = os.path.dirname(__file__) + "/../../thread_2/"
    # inp_folder = "/media/emmah/PortableSSD/Arclab_data/meat_thread_data_9_26/"
    inp_folder = "/media/arclab/PortableSSD/Arclab_data/4_01_26/"
    trial = "trial_1"
    # prefixes = ["left_recif_", "right_recif_"]
    prefixes = ["left_", "right_"]
    left_right_folder = ["left_rgb/", "right_rgb/"]
    start = 0
    ext = ".png" # ".jpg"
    # calib = os.path.dirname(__file__) + "/../../camera_calibration_sarah.yaml"
    # calib = os.path.dirname(__file__) + "/../assets/camera_calibration_sarah.yaml"
    calib = os.path.dirname(__file__) + "/../assets/camera_calibration_fei.yaml"
    if USE_SAM:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = "vit_h"
        segmenter = SAMSegmenter(device, model_type)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter = UNetSegmenter(device)
    for i in range(0, 200, 50): #end at 279
        print(start+i)
        imfile1 = inp_folder+trial+"/"+left_right_folder[0]+trial+"_"+"{:03d}".format(start+i)+ext
        img1 = cv2.imread(imfile1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # imfile2 = inp_folder+prefixes[1]+str(start+i)+ext
        imfile2 = inp_folder+trial+"/"+left_right_folder[1]+trial+"_"+"{:03d}".format(start+i)+ext
        img2 = cv2.imread(imfile2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        try:
            fit_eval(img1, img2, calib, segmenter)
        except Exception as e:
            print("FAILED: " + str(e))