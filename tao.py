import cv2
from real_env import get_detic_predictions, get_segmentation_mask, get_bb_labels
import torch
from segment_anything import sam_model_registry, SamPredictor


fname = "/Users/meenalp/Downloads/image_0.png"
img = cv2.imread(fname)[:,:,::-1]


pred_lst, names = get_detic_predictions(
    [img], vocabulary="custom", custom_vocabulary="mango,banana,mug,tray"
)


sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

print("Obtaining segmentation mask")
pred_boxes, pred_labels = get_bb_labels(pred_lst[0], names)

# sam_predictor.set_image(img, image_format="RGB")
# sam_predictions_dir = "sam_predictions"

# masks, scores, logits = sam_predictor.predict(multimask_output=False)

seg, embedding_dict = get_segmentation_mask(
    sam_predictor,
    img,
    None,
    pred_boxes,
    pred_labels,
    prefix=0,
)