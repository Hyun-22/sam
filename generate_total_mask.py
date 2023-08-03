
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import cv2
import sys
from glob import glob
sys.path.append("..")
# from segment_anything import sam_model_registry, SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
if __name__ == "__main__":
    
    start = time.time()
    
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    
    device = "cuda:0"
    model_type = "vit_h"
    
    # whole scan image
    # image_paths = glob("/home/ailab/AILabDataset/01_Open_Dataset/19_ViewofDelft/vod_full/camera/*.jpg")
    
    # cropped image
    image_paths = glob("/home/ailab/AILabDataset/01_Open_Dataset/19_ViewofDelft/vod_object/train/camera/*.png")
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    end = time.time()
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print("model setting done")
    print("using time for loading model: ", end-start)
    for image_path in image_paths:
        start = time.time()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        end = time.time()
        print("using time for inference: ", end-start)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show() 
    
    # predictor = SamPredictor(sam)
    # predictor.set_image(image)
    
    # input_point = np.array([[630, 550]])
    # input_label = np.array([1])
    
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True,
    # )
    
    # for i, (mask, score) in enumerate(zip(masks, scores)):
        
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()  