import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
model_type = "vit_h"
# device = "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image_fnames = os.listdir("current_images")

from real_env import get_detic_predictions

pred_lst, names = get_detic_predictions(None)


sam_predictions_dir = "sam_predictions"
os.makedirs(sam_predictions_dir, exist_ok=True)

for fname in image_fnames:
    if fname.endswith('.png'):
        im = cv2.imread(os.path.join("current_images", fname))
        predictor.set_image(im, image_format='RGB')

        idx = int(fname[0])
        bbs = pred_lst[idx]["instances"].pred_boxes.tensor.cpu().numpy()

        for j in range(len(bbs)):
            masks, scores, logits = predictor.predict(
                box=bbs[j],
                multimask_output=False
            )

            plt.figure(figsize=(10, 10))
            plt.imshow(im)
            show_mask(masks[0], plt.gca())
            show_box(bbs[j], plt.gca())
            plt.axis('off')

            plt.savefig(sam_predictions_dir + f"/{fname}_{j}.png")

