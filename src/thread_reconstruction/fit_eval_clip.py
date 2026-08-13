import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.interpolate as interp
import cv2
import os
import torch
import pickle
import pdb

from src.thread_reconstruction.segmenter_clip import SAMSegmenter, UNetSegmenter, HandSegmenter
from src.thread_reconstruction.keypt_selection import keypt_selection
from src.thread_reconstruction.keypt_ordering import keypt_ordering
from src.thread_reconstruction.optim import optim
from src.thread_reconstruction.utils import *
from src.thread_reconstruction.hand_keypoints import keypt_hand_select, hand_ordering
import argparse


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
    new_size = (640, 480) # ok on 9_26 data
    # new_size = (640, 360) # fix for  11_17 data
    # new_size = (1920, 1080) # fix for 11_17 data
    

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
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, Q)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
    spline, spline_specs = optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, cam2img, P1, P2)

    return spline, mask1, mask2


def fit_eval_hand_seg(img1, img2, img1_t_mask, img2_t_mask, img1_n_mask, img2_n_mask, calib, segmenter):
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
    new_size = (640, 480) # ok on 9_26 data
    # new_size = (640, 360) # fix for  11_17 data
    # new_size = (1920, 1080) # fix for 11_17 data
    

    # Rectify image and store necessary matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
    
    cam2img = P1[:,:-1]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, new_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, new_size, cv2.CV_16SC2)
    # skip rectifying image again
    # img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    # img1_mask = cv2.remap(img1_mask, map1x, map1y, cv2.INTER_LINEAR)
    # img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    # img2_mask = cv2.remap(img2_mask, map2x, map2y, cv2.INTER_LINEAR)

    # Segment stereo images
    mask1_t = cv2.cvtColor(img1_t_mask, cv2.COLOR_RGB2GRAY)
    mask2_t = cv2.cvtColor(img2_t_mask, cv2.COLOR_RGB2GRAY)
    mask1_n = cv2.cvtColor(img1_n_mask, cv2.COLOR_RGB2GRAY)
    mask2_n = cv2.cvtColor(img2_n_mask, cv2.COLOR_RGB2GRAY)

    mask1 = mask1_t+mask1_n
    mask2 = mask2_t+mask2_n
    stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
    img1 = np.where(stack_mask1>0, img1, 0)
    stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
    img2 = np.where(stack_mask2>0, img2, 0)
    
    # Convert from btyes to float
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # # Perform reconstruction
    # Q = np.array([[1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.6791902e+02], 
    #               [0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 2.3415271e+02], 
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0258822e+03], 
    #               [0.0000000e+00, 0.0000000e+00, 1.46621652e-01, 0.0000000e+00]])
    # '''
    # [[ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -1.67919014e+02]
    # [ 0.00000000e+00  1.00000000e+00  0.00000000e+00 -2.34152710e+02]
    # [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.02588223e+03]
    # [ 0.00000000e+00  0.00000000e+00  1.46621652e-01 -0.00000000e+00]]
    # '''


    # hand pick points
    hand_pick = 2
    if hand_pick == 1:
        img_3D, keypoints, grow_paths, order = keypt_hand_select(img1, img2, mask1, mask2, Q)
        # optim actually only needs masks, keypoints, order, cam2img, p1, p2

    if hand_pick == 2:
        img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, Q)
        img_3D, keypoints, grow_paths, order = hand_ordering(img1, img_3D, cluster_map, keypoints, grow_paths, adjacents, needle_pos_file, P1) # clusters is not used in keypt_ordering, hence removed


    else: 
    # neelay's code
        img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, Q)
        img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)


    spline, spline_specs = optim(img1, mask1_t, mask2_t, mask1_n, mask2_n, img_3D, keypoints, grow_paths, order, cam2img, P1, P2, needle_pos_file)
    return spline, spline_specs, mask1, mask2

def union_mask(masks:list):
    output = masks[0]
    output = output[(masks[0] == 225) & (masks[1] == 225)]
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_spline', '-s', help="store list of splines to file [True, False]", action='store_true')
    parser.add_argument('--spline_nodes', '-n', default=20, help="number of spline points to save")
    parser.add_argument('--save_mask', action='store_true', help='save needle mask to file')
    args = parser.parse_args()
                        
    USE_SAM = True # set true to use sam
    USE_HAND = True
    # inp_folder = os.path.dirname(__file__) + "/../../thread_2/"
    data_select = "3_21_collected"
    if data_select == "9_26": #9_26 data
        inp_folder = "/media/emmah/PortableSSD/Arclab_data/meat_thread_data_9_26/trial"
        trial = "trial_06" #5, 7, 8, 9, 10
        lr_folder = [f"/left_rgb/{trial}_left_", f"/right_rgb/{trial}_right_"]
        prefixes = [f"{trial}_left_", f"{trial}_right_"] # 9_26 data
        # file_l = inp_folder+trial+"/"+lr_folder[0]+prefixes[0]
        # file_r = inp_folder+trial+"/"+lr_folder[1]+prefixes[1]
        ext = ".png"

    elif data_select == "11_17": # 11_17 requires fixing resizing above, image is unrectified.
        trial = "trial_06" #5, 7, 8, 9, 10
        inp_folder = "/media/emmah/PortableSSD/Arclab_data/meat_thread_data_11_17/"
        lr_folder = ["/left_rgb/frame_", "/right_rgb/frame_"] 
        ext = ".png"

    elif data_select == "thread_2": # Neelay data thread_2
        inp_folder = "/media/emmah/PortableSSD/Arclab_data/thread_2/"
        lr_folder = ["left_rgb/left_recif_", "right_rgb/right_recif_"] 
        trial = "thread_2" #5, 7, 8, 9, 10
        prefixes = [f"{trial}_left_", f"{trial}_right_"] # for saving output spline
        # file_l = inp_folder+lr_folder[0]
        # file_r = inp_folder+lr_folder[1]
        ext = ".jpg"

    elif data_select == "3_21":
        trial = 20
        inp_folder = f"/media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/trial_" + f"{trial}"
        lr_folder = ["/left_rgb/frame_", "/right_rgb/frame_"]
        trial = "trial" + f"{trial}" #5, 7, 8, 9, 10
        prefixes = [f"{trial}_left_", f"{trial}_right_"] # for saving output spline
        ext = ".png"

    elif data_select == "3_21_collected":
        trial = 28
        n_mask = True

        inp_folder = f"/media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/thread_meat_3_21_collected"
        # lr_folder = [f"/trial_{trial}_left", f"/trial_{trial}_right"]
        lr_folder = ["/","/"] # folders are defined after use selection below
        # trial = "trial" + f"{trial}" #5, 7, 8, 9, 10
        # prefixes = [f"{trial}_left_", f"{trial}_right_"] # for saving output spline
        ext = ".png"


    elif data_select == "3_21_edited":
        trial = 20
        inp_folder = f"/media/emmah/PortableSSD/Arclab_data/thread_meat_3_21/edited_photos"
        lr_folder = [f"/trial_{trial}_left", f"/trial_{trial}_right"]
        trial = "trial" + f"{trial}" #5, 7, 8, 9, 10
        prefixes = [f"{trial}_left_", f"{trial}_right_"] # for saving output spline
        ext = ".png"

    file_l = inp_folder+lr_folder[0]
    file_r = inp_folder+lr_folder[1]
    start = 0

    splines = []
    masks = []
    # calib = os.path.dirname(__file__) + "/../../camera_calibration_sarah.yaml"
    # calib = os.path.dirname(__file__) + "/../assets/camera_calibration_sarah.yaml"
    calib = os.path.dirname(__file__) + "/../assets/camera_calibration_fei.yaml"
    # calib = os.path.dirname(__file__) + "/../assets/dvrk_camera_calibration_before_rectify.yaml"
    if USE_SAM:
        device = torch.device('cpu')
        model_type = "vit_h"
        segmenter = SAMSegmenter(device, model_type)
    elif USE_HAND:
        device = torch.device('cpu')
        segmenter = HandSegmenter(device) # placeholder, doesn't do anything
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter = UNetSegmenter(device)


    def count_images_in_folder(folder_path):
        image_count = 0
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                image_count += 1
        return image_count
    
    image_count = count_images_in_folder(os.path.dirname(file_l))

    sampling = False
    if not sampling: # user input
        while True:

            user_sample = input(f"choose frame 0-{image_count-1}: ")
            print(f"input: {user_sample}")

            if user_sample.isdigit():
                if int(user_sample) >= image_count:
                    break
                i = int(user_sample)
            elif user_sample.lower == "quit" or "q":
                print("exiting after user input")
                break
            else:
                print("invalid input") 

            if data_select == "thread_2":
                print("file", f"{file_l}{start+i}{ext}")
                
            elif data_select == "3_21_edited" or data_select == "3_21_collected":
                print("file", f"{file_l}{ext}")
        
            else:
                print("file", file_l+"{:06d}".format(start+i)+ext)

            if data_select == "thread_2": # different file formating
                imfile1 = file_l+f"{start+i}"+ext
                imfile2 = file_r+f"{start+i}"+ext

            elif data_select == "3_21_edited":
                imfile1 = file_l+ext
                imfile2 = file_r+ext
                if USE_HAND:
                    img1mask = file_l + "_mask" + ext
                    img2mask = file_r + "_mask" + ext

            elif data_select == "3_21_collected":
            
                trial = i
                lr_folder = [f"/trial_{trial}_left", f"/trial_{trial}_right"]
                prefixes = [f"{trial}_left_", f"{trial}_right_"] # for saving output spline
                file_l = inp_folder + lr_folder[0]
                file_r = inp_folder + lr_folder[1]
                imfile1 = file_l + ext
                imfile2 = file_r + ext
                if USE_HAND:
                    img1_t_mask = file_l + "_mask" + ext
                    img2_t_mask = file_r + "_mask" + ext
                    img1_n_mask = file_l + "_n_mask" + ext if n_mask else img1_t_mask
                    img2_n_mask = file_r + "_n_mask" + ext if n_mask else img2_t_mask

            else:
                imfile1 = file_l+"{:06d}".format(start+i)+ext
                imfile2 = file_r+"{:06d}".format(start+i)+ext
                
            img1 = cv2.imread(imfile1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # imfile2 = inp_folder+prefixes[1]+str(start+i)+ext
            img2 = cv2.imread(imfile2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            if USE_HAND:
                img1_t_mask = cv2.cvtColor(cv2.imread(img1_t_mask), cv2.COLOR_BGR2RGB)
                img1_n_mask = cv2.cvtColor(cv2.imread(img1_n_mask), cv2.COLOR_BGR2RGB)
                img2_t_mask = cv2.cvtColor(cv2.imread(img2_t_mask), cv2.COLOR_BGR2RGB)
                img2_n_mask = cv2.cvtColor(cv2.imread(img2_n_mask), cv2.COLOR_BGR2RGB)

            # try:
            if USE_HAND:
                spline, spline_specs, mask1, mask2 = fit_eval_hand_seg(img1, img2, img1_t_mask, img2_t_mask, img1_n_mask, img2_n_mask, calib, segmenter)
            else:
                spline, mask1, mask2 = fit_eval(img1, img2, calib, segmenter)

            save_spline = input(f"save spline from this trial? y/n ")

            # if args.output_spline == True:
            if save_spline == 'y':
                    with open(inp_folder + f"/trial_{trial}_" + "spline.npy", "wb") as f:
                        print("saving", f"trial {trial}", "spline")
                        # np.save(f, np.array(spline(np.linspace(0, 1, 80))))
                        np.save(f, np.array(spline))

                    with open(inp_folder + f"/trial_{trial}_" + "spline_specs.pkl", "wb") as f:
                        print("saving", f"trial {trial}", "spline specs")
                        pickle.dump(spline_specs, f)

    if args.save_mask == True:
        i = 0
        print("saving", len(masks), "mask(s)")
        for mask in masks:
            cv2.imwrite(inp_folder + trial + "/" + "mask_" + prefixes[i%2]+ "{:06d}".format(i-(i%2))+ext, mask)
            i+=1