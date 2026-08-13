import cv2
import torch
import numpy as np
import os

class Segmenter:
    def __init__(self, device):
        self.device = device

    def segmentation(self, img):
        raise NotImplementedError
    
# This is for a UNet I overfit to another dataset. Feel free to ignore
class UNetSegmenter(Segmenter):
    def __init__(self, device):
        super().__init__(device)
        self.unet = torch.load(os.path.dirname(__file__) + "/../segmenter.pt")
        self.unet.to(device=self.device)

    def segmentation(self, img):
        inp = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        inp = torch.from_numpy(inp).to(device=self.device)
        inp = inp.permute(2,0,1)
        out = self.unet(inp.unsqueeze(0))
        mask = out.squeeze().detach().to(device="cpu").numpy()
        mask = np.where(mask>=0.5, 1, 0)
        return mask

class HandSegmenter(Segmenter):
    def __init__(self, device):
        pass
        # super().__init__(device)

    def segmentation(self, img):
        mask = None
        # inp = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # inp = torch.from_numpy(inp).to(device=self.device)
        # inp = inp.permute(2,0,1)
        # out = self.unet(inp.unsqueeze(0))
        # mask = out.squeeze().detach().to(device="cpu").numpy()
        # mask = np.where(mask>=0.5, 1, 0)
        return mask

# This is what you should use
# device: either "cpu" or "cuda"
# model_type: define size of SAM_HQ encoder, best is "vit_h"
class SAMSegmenter(Segmenter):
    def __init__(self, device, model_type="vit_h"):
        super().__init__(device)
        from segment_anything import sam_model_registry, SamPredictor

        if model_type == "vit_h": 
            sam_checkpoint = "/home/emmah/ARClab/segment-anything/segment_anything/checkpoint/sam_hq_vit_h.pth"
        elif model_type == "vit_l":
            sam_checkpoint = "/home/emmah/ARClab/segment-anything/segment_anything/checkpoint/sam_vit_l_0b3195.pth"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)


    def segmentation(self, image):
        print("Generating embedding")

        self.predictor.set_image(image)
        clone = image.copy()

        print("Embedding generated!")

        clip_box = None
        clip_box = np.array((0, 0, image.shape[1], image.shape[0])) # default clip box is the entire image
        clip = True # rectangle clipping mode
        print("clip is", clip)
        ix, iy = None, None # inital x, inital y
        drawing = False # true when mouse is pressed

        points = []
        labels = []
        masks = None
        w_name = "Add (L-click) and Remove (R-click) Mask. C to toggle clip box. Esc to Quit"

        # Runs segmentation window
        def clip_and_select_point(event, x, y, flags, param):
            nonlocal points, labels, masks, ix, iy, clip, drawing, clone, clip_box
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                if clip is True:
                    ix, iy = x, y 

            if event == cv2.EVENT_RBUTTONDOWN:
                # if (ix, iy)==(x, y):
                #     clip_box = None
                # else:
                #     clip_box = np.array(ix, iy, x, y)
                points.append((x, y))
                labels.append(0)
                masks, _, _ = self.predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array(labels),
                    box=clip_box[None, :],
                    multimask_output=False,
                )
                clone = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
                if clip_box is not None:
                    cv2.rectangle(clone, (clip_box[0:2]), (clip_box[2:4]), (0, 255, 0), 1)
                if points is not None:
                    clone[masks[0]>0] = np.array([255, 144, 33])
                    for point, label in zip(points, labels):
                        cv2.circle(clone, point, 5, 
                                (0,255,0) if label==1 else (0,0,255), 2)
                cv2.imshow(w_name, clone)

            if event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if clip == True:
                    clip_box = np.array((ix, iy, x, y))
                    # cv2.rectangle(clone, (ix,iy), (x, y), (0, 255, 0), 1)

                    # clone = image.copy()
                    print("clip box set to", clip_box)
                        

                else:
                    points.append((x, y))
                    labels.append(1)
                    masks, _, _ = self.predictor.predict(
                        point_coords=np.array(points),
                        point_labels=np.array(labels),
                        box=clip_box[None, :],
                        multimask_output=False,
                    )
                clone = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
                if clip_box is not None:
                    cv2.rectangle(clone, (clip_box[0:2]), (clip_box[2:4]), (0, 255, 0), 1)
                else:
                    print("clip box is None")
                if masks is not None:
                    clone[masks[0]>0] = np.array([255, 144, 33])
                    for point, label in zip(points, labels):
                        cv2.circle(clone, point, 5, 
                                (0,255,0) if label==1 else (0,0,255), 2)
                cv2.imshow(w_name, clone)


            

            # if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN: 
            #     points.append((x, y))
            #     labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            #     masks, _, _ = self.predictor.predict(
            #         point_coords=np.array(points),
            #         point_labels=np.array(labels),
            #         box=clip_box,
            #         multimask_output=False,
            #     )
            #     clone = image.copy()
            #     clone = cv2.cvtColor(clone, cv2.COLOR_RGB2BGR)
            #     clone[masks[0]>0] = np.array([255, 144, 33])
            #     for point, label in zip(points, labels):
            #         cv2.circle(clone, point, 5, 
            #                 (0,255,0) if label==1 else (0,0,255), 2)
            #     # cv2.imshow(w_name, clone)
            
        cv2.namedWindow(w_name)
        cv2.setMouseCallback(w_name, clip_and_select_point)
        # clone = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        cv2.imshow(w_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # clone = image.copy()
        while(1):
            # cv2.imshow(w_name, cv2.cvtColor(clone, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c'):
                clip = not clip
                print("clip is", clip)
            elif k == 27: # esc
                break
        cv2.destroyAllWindows()
        return masks[0]*255