import cv2
import torch
import numpy as np
import os
# from segment_anything import sam_model_registry, SamPredictor

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

# This is what you should use
# device: either "cpu" or "cuda"
# model_type: define size of SAM_HQ encoder, best is "vit_h"
# class SAMSegmenter(Segmenter):
#     def __init__(self, device, model_type="vit_h"):
#         super().__init__(device)
    
#         if model_type == "vit_h": 
#             sam_checkpoint = "/home/emmah/ARClab/segment-anything/segment_anything/checkpoint/sam_hq_vit_h.pth"
#         elif model_type == "vit_l":
#             sam_checkpoint = "/home/emmah/ARClab/segment-anything/segment_anything/checkpoint/sam_vit_l_0b3195.pth"

#         sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=self.device)
#         self.predictor = SamPredictor(sam)

#     def segmentation(self, image):
#         print("Generating embedding")

#         self.predictor.set_image(image)

#         print("Embedding generated!")

#         points = []
#         labels = []
#         masks = None
#         w_name = "Add (L-click) and Remove (R-click) Mask. Esc to Quit"

#         # Runs segmentation window
#         def select_point(event, x, y, flags, param):
#             nonlocal points, labels, masks
#             if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN: 
#                 points.append((x, y))
#                 labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
#                 masks, _, _ = self.predictor.predict(
#                     point_coords=np.array(points),
#                     point_labels=np.array(labels),
#                     box=None,
#                     multimask_output=False,
#                 )
#                 clone = image.copy()
#                 clone = cv2.cvtColor(clone, cv2.COLOR_RGB2BGR)
#                 clone[masks[0]>0] = np.array([255, 144, 33])
#                 for point, label in zip(points, labels):
#                     cv2.circle(clone, point, 5, 
#                             (0,255,0) if label==1 else (0,0,255), 2)
#                 cv2.imshow(w_name, clone)
            
#         cv2.namedWindow(w_name)
#         cv2.setMouseCallback(w_name, select_point)
#         cv2.imshow(w_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         return masks[0]*255