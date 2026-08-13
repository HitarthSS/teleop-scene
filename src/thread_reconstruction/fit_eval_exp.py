import cProfile
import pstats
# from pstats import SortKey
from io import StringIO
pr = cProfile.Profile()
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
# import scipy.interpolate as interp
import cv2

import os
import pickle
import pdb
from pathlib import Path

from src.thread_reconstruction.keypt_selection import keypt_selection
from src.thread_reconstruction.keypt_ordering import keypt_ordering
from src.thread_reconstruction.warm_start_keypoints import WarmStart
from src.thread_reconstruction.optim import optim, optim_warm_start, Optim
# from utils import *
from src.thread_reconstruction.hand_keypoints import hand_ordering
import argparse
import traceback
import sys

# Drop into pdb at every error / degenerate return in this module, so a
# silently-skipped frame stops instead of being swallowed by the caller's
# fallback path.  Export THREAD_RECON_BREAK=0 to disable every breakpoint
# in the package at once (unattended runs, live ROS sessions).
DEBUG_BREAK_ON_ERROR = os.environ.get("THREAD_RECON_BREAK", "1") != "0"

"""
img1: RGB left camera image (np array)
img2: RGB right camera image (np array)
calib: filename for camera calibration file (string)
segmenter: segmentation object (see segmenter.py)
"""
class FitEvalClass():
    def __init__(self, 
                 args
                 ):
        print('python version in fit script')
        print(sys.version)

        self.trial_path=args.trial_path
        self.trial_name=args.trial_name
        self.speedy = args.speedy
        self.ros_enable=args.ros_enable
        self.hand_order=args.hand_order
        if args.calib==None:
            # inp_folder = os.path.dirname(__file__) + "/../../thread_2/"
            # inp_folder = f"/media/emmah/PortableSSD/Arclab_data/{trial_path}/{trial_name}/"
            # inp_folder = f"/media/arclab/PortableSSD/Arclab_data/{parent_folder}/{trial_name}/"
            # calib = os.path.dirname(__file__) + "/../../camera_calibration_sarah.yaml"
            # calib = os.path.dirname(__file__) + "/../assets/camera_calibration_sarah.yaml"
            calib = os.path.dirname(__file__) + "/../assets/camera_calibration_fei.yaml"
        else:
            calib = args.calib
            
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
        print("cv_file img_size", img_size)
        new_size = (640, 480) # ok on 9_26 data

        # Rectify image and store necessary matrices
        # R1, R2, P1, P2, Q, roi1, roi2

        _, _, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
        
        self.cam2img = self.P1[:,:-1]        


    def fit_eval_hand_seg(self, 
                          frame,
                          warm_start_frame=None,
                          needle_mask=None,
                          spline_file=None
                          ):
        # pr.enable()
        if needle_mask is None:
            needle_pos_file=None
        lr_folder = ["left_rgb/", "right_rgb/"]
        ext = ".png"

        # n_mask = True if needle_mask is not None else False

        imfile1 = self.trial_path+lr_folder[0]+self.trial_name+"_{:03d}".format(frame)+ext
        imfile2 = self.trial_path+lr_folder[1]+self.trial_name+"_{:03d}".format(frame)+ext
        print(f"img1 path {imfile1}")
        # needle_pos_file = f"{self.trial_path}/needle_pose/{self.trial_name}_needle_pose.pkl"
        # needle_pos_file = f"/media/emmah/PortableSSD/Arclab_data/{self.trial_path}/{self.trial_name}/needle_pose/trial_{trial}_needle_pose.pkl"
        # needle_pos_file = f"/media/arclab/PortableSSD/Arclab_data/{parent_folder}/{self.trial_name}/needle_pose/trial_{trial}_needle_pose.pkl"
        # if Path(needle_pos_file).exists():
        #     print(f"needle pos file at {needle_pos_file}")
        # else:
        #     needle_pos_file = None
        #     print(f"needle pos file not found")

        t_mask_1 = Path(self.trial_path + lr_folder[0]+self.trial_name+"_{:03d}".format(frame) + "_mask" + ext)
        t_mask_2 = Path(self.trial_path + lr_folder[0]+"binary_masks/" + self.trial_name + "_{:03d}".format(frame) + ext)
        if t_mask_1.exists():
            img2_t_mask = self.trial_path + lr_folder[1] + self.trial_name + "_{:03d}".format(frame) + "_mask" + ext
            img1_t_mask = self.trial_path + lr_folder[0] + self.trial_name + "_{:03d}".format(frame) + "_mask" + ext
        elif t_mask_2.exists():
            img1_t_mask = self.trial_path + lr_folder[0]+"binary_masks/" + self.trial_name + "_{:03d}".format(frame) + ext
            img2_t_mask = self.trial_path + lr_folder[1]+"binary_masks/" + self.trial_name + "_{:03d}".format(frame) + ext

        else:
            raise FileNotFoundError(f"File not found in either location:\n{t_mask_1}\n{t_mask_2}")

        # img1_n_mask = self.trial_path+lr_folder[0] + self.trial_name + "_{:03d}".format(frame) + "_n_mask" + ext if n_mask else img1_t_mask
        # img2_n_mask = self.trial_path+lr_folder[1] + self.trial_name + "_{:03d}".format(frame) + "_n_mask" + ext if n_mask else img2_t_mask

        img1_t_mask = cv2.cvtColor(cv2.imread(img1_t_mask), cv2.COLOR_BGR2RGB)
        # img1_n_mask = cv2.cvtColor(cv2.imread(img1_n_mask), cv2.COLOR_BGR2RGB)
        img2_t_mask = cv2.cvtColor(cv2.imread(img2_t_mask), cv2.COLOR_BGR2RGB)
        # img2_n_mask = cv2.cvtColor(cv2.imread(img2_n_mask), cv2.COLOR_BGR2RGB)

        img1 = cv2.imread(imfile1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(imfile2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        thread_file_path = self.trial_path + self.trial_name+"_{:03d}_".format(frame)+"spline.pkl"
        warm_thread_path = self.trial_path + self.trial_name+"_{:03d}_".format(warm_start_frame)+"spline.pkl"
        thread_specs_path = self.trial_path + self.trial_name+"_{:03d}_".format(frame) + "spline_specs.pkl"
        warm_thread_specs_path = self.trial_path + self.trial_name+"_{:03d}_".format(warm_start_frame) + "spline_specs.pkl"

        # Segment stereo images
        mask1_t = cv2.cvtColor(img1_t_mask, cv2.COLOR_RGB2GRAY)
        mask2_t = cv2.cvtColor(img2_t_mask, cv2.COLOR_RGB2GRAY)
        mask1_n, mask2_n = None, None
        # mask1_n = cv2.cvtColor(img1_n_mask, cv2.COLOR_RGB2GRAY)
        # mask2_n = cv2.cvtColor(img2_n_mask, cv2.COLOR_RGB2GRAY)

        # mask1 = mask1_t+mask1_n
        # mask2 = mask2_t+mask2_n
        mask1 = mask1_t
        mask2 = mask2_t
        stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
        img1 = np.where(stack_mask1>0, img1, 0)
        stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
        img2 = np.where(stack_mask2>0, img2, 0)
        
        # Convert from btyes to float
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # print("fit eval cv2 version", cv2.__version__)
        # print(f"img1 shape {img1.shape}, mask1 shape {mask1.shape}")
        # merged = cv2.hconcat([img1, img2])
        # cv2.imshow("img1, img2 of hand seg", merged)
        # cv2.imshow("mask1 of hand seg", mask1)
        # cv2.waitKey(0)

        img_3D, __, cluster_map, keypoints, grow_paths, adjacents =  keypt_selection(img1, img2, mask1, self.Q)

        Warm = WarmStart()
        keypoints, order, warm_keypts = Warm.warm_start_keypoints(mask1, keypoints, 
                                                            spline_file=spline_file,
                                                            P1=self.P1, 
                                                            speedy=self.speedy, 
                                                            )
        if keypoints is None: # if warm start keypoint matching fails
            if self.hand_order:
                keypoints, order = hand_ordering(img1, img_3D, keypoints, needle_pos_file, self.P1) # clusters is not used in keypt_ordering, hence removed
            else: # original method auto ordering
                __, keypoints, __, order = keypt_ordering(img1, img_3D, cluster_map, keypoints, grow_paths, adjacents)

            spline, spline_specs = optim(mask1_t, mask2_t, mask1_n, mask2_n, 
                                        keypoints, 
                                        order, 
                                        self.cam2img, 
                                        self.P1, self.P2, 
                                        needle_pos_file)

        else:
            spline, spline_specs = optim_warm_start(mask1_t, mask2_t,
                                                    mask1_n, mask2_n,
                                                    keypoints,
                                                    order,
                                                    self.cam2img,
                                                    self.P1,
                                                    spline_file=spline_file,
                                                    warm_start_keypts=warm_keypts,
                                                    speedy=self.speedy,
                                                    ros_enable=self.ros_enable,
                                                    needle_pos_file=needle_pos_file
                                                    )
        # pr.disable()
            if not self.hand_order: # only check for orientation matching when not hand ordering
                spline, spline_specs = Optim().match_warm_order(spline, spline_specs, spline_file=self.spline_file)
        return spline, spline_specs
    
    def seek_warm_start(self, frame):
        warm_thread_path = self.trial_path + self.trial_name+"_{:03d}_".format(frame)+"spline.pkl"
        if Path(warm_thread_path).exists():
            warm_specs_path = self.trial_path + self.trial_name+"_{:03d}_".format(frame) + "spline_specs.pkl"
            print(f"warm start frame: {frame}\n")
            return warm_thread_path, warm_specs_path
        elif frame==0:
            print("Frame 0 reached and no warm start found\n")
            if DEBUG_BREAK_ON_ERROR: pdb.set_trace()
            return None, None

        return self.seek_warm_start(frame-1)

    def save_spline(self, spline, spline_specs, frame):
        save_spline = input(f"save spline from this trial? (y) ")
        # save_spline = 'n'

        if save_spline == 'y':
            # with open(inp_folder + self.trial_name+"_{:03d}_".format(frame)+"spline.npy", "wb") as f:
            with open(self.spline_file, "wb") as f:
                print("saving", f"{self.trial_name}", "spline\n")
                # np.save(f, np.array(spline(np.linspace(0, 1, 80))))
                # np.save(f, np.array(spline))
                pickle.dump(spline, f) # pickle dump for a bspline object.

            with open(self.spline_specs_file, "wb") as f:
                print("saving", f"{self.trial_name}", "spline specs\n")
                pickle.dump(spline_specs, f)
        
    def main(self, frame, warm_start_frame, needle_mask=None):
        self.spline_file = self.trial_path + self.trial_name+"_{:03d}_".format(frame)+"spline.pkl"
        self.spline_specs_file = self.trial_path + self.trial_name+"_{:03d}_".format(frame) + "spline_specs.pkl"

        spline, spline_specs = self.fit_eval_hand_seg(frame=frame,
                                                      warm_start_frame=warm_start_frame,
                                                      needle_mask=needle_mask,
                                                      spline_file=self.spline_file
                                                      )


        self.save_spline(spline, spline_specs, frame=frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_spline', '-s', help="store list of splines to file [True, False]", action='store_true')
    # parser.add_argument('--spline_nodes', '-n', type=int, default=20, help="number of spline points to save")
    # parser.add_argument('--save_mask', action='store_true', help='save needle mask to file')
    # parser.add_argument('--segmenter', default='hand', help="sam, hand, or unet")
    # parser.add_argument('--needle_mask', default=None)
    parser.add_argument('--trial_path', help="full path to the directory of the trial used")
    parser.add_argument('--trial_name', help='trial name is usually trial_xx or trial_xx_video')
    parser.add_argument('--frame', type=int, default=1, help='choose specific frame in trial to test')
    parser.add_argument('--warm_start_frame', '--w', type=int, default=None, help='choose the frame of the warm start spline')
    parser.add_argument('--speedy', action="store_true")
    parser.add_argument('--ros_enable', action='store_true')
    parser.add_argument('--calib', default=None, help="pass in camera calibration file path")
    # hand order or neelay order
    parser.add_argument('--hand_order', action='store_true', help='set to true to hand order')
    args = parser.parse_args()
                        
    fit_eval = FitEvalClass(args
                 )
    try:
        fit_eval.main(frame=args.frame,
             warm_start_frame=args.warm_start_frame,
             )
        
    except Exception as e:
        print(f"Caught error:{e}")
        print("Traceback (most recent call last):")
        traceback.print_exc()

    # p = pstats.Stats('restats')
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats("time") # cumulative
    # ps.print_stats(20)  # top 20
    # print(s.getvalue())

