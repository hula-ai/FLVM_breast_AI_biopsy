import matplotlib.pyplot as plt
import numpy as np
import torch
import PIL
import aim
import cv2

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def matplotlib_imshow(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = torch.permute(img, (1, 2, 0))
    img = img*np.array(std) + np.array(mean)     # unnormalize
    npimg = img.cpu().numpy()

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(npimg)


def vis_gradcam_mamm(dataloader, model, target_layers, device, num_log_images):
    color_dict = {
        'antiquewhite': (250,235,215),
        'aqua': (0,255,255),
        'cadmiumorange': (255,97,3),
        'crimson': (220,20,60),
        'darkseagreen': (143,188,143),
        'khaki': (240,230,140),
        'lightskyblue': (135,206,250),
        'thistle': (216,191,216),
        'violet': (238,130,238),
        'olivedrab': (107,142,35),

        'antiquewhite': (250,235,215),
        'aqua': (0,255,255),
        'cadmiumorange': (255,97,3),
        'crimson': (220,20,60),
        'darkseagreen': (143,188,143),
        'khaki': (240,230,140),
        'lightskyblue': (135,206,250),
        'thistle': (216,191,216),
        'violet': (238,130,238),
        'olivedrab': (107,142,35),

        'antiquewhite': (250,235,215),
        'aqua': (0,255,255),
        'cadmiumorange': (255,97,3),
        'crimson': (220,20,60),
        'darkseagreen': (143,188,143),
        'khaki': (240,230,140),
        'lightskyblue': (135,206,250),
        'thistle': (216,191,216),
        'violet': (238,130,238),
        'olivedrab': (107,142,35),

        'antiquewhite': (250,235,215),
        'aqua': (0,255,255),
        'cadmiumorange': (255,97,3),
        'crimson': (220,20,60),
        'darkseagreen': (143,188,143),
        'khaki': (240,230,140),
        'lightskyblue': (135,206,250),
        'thistle': (216,191,216),
        'violet': (238,130,238),
        'olivedrab': (107,142,35),
    }
    colors = list(color_dict.keys())

    aim_viss = []
    total_images = 0
    for step, batch_data in enumerate(dataloader): 
        if total_images == num_log_images:
            break

        mamm, label = batch_data['image'], batch_data['label']

        mamm = mamm.to(device)
        label = label.to(device)

        pos_targets = [ClassifierOutputTarget(1)] * mamm.shape[0]
        neg_targets = [ClassifierOutputTarget(0)] * mamm.shape[0]

        cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)
        
        pos_grayscale_cam = cam(input_tensor=mamm, targets=pos_targets)
        neg_grayscale_cam = cam(input_tensor=mamm, targets=neg_targets)

        for idx in range(mamm.shape[0]):
            if total_images == num_log_images:
                break
            
            # read mammogram
            mamm_path = batch_data['image_path'][idx]
            mamm_img = cv2.imread(mamm_path)
            mamm_img = cv2.resize(mamm_img, (448, 576), interpolation=cv2.INTER_LINEAR)
            
            # Draw CAM
            mamm_pos_grayscale_cam = pos_grayscale_cam[idx, :]
            mamm_neg_grayscale_cam = neg_grayscale_cam[idx, :]

            pos_visualization = show_cam_on_image(mamm_img.astype(np.float32)/255, mamm_pos_grayscale_cam, use_rgb=True)
            neg_visualization = show_cam_on_image(mamm_img.astype(np.float32)/255, mamm_neg_grayscale_cam, use_rgb=True)

            # Draw findings annotations
            for abn_idx, mask_path in enumerate(batch_data['mask_path'][idx]):
                if not mask_path:
                    break
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.resize(mask_img, (448, 576), interpolation=cv2.INTER_LINEAR)
                
                
                ret,thresh = cv2.threshold(mask_img,127,255,0)
                contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                cv2.drawContours(mamm_img, contours, -1, color_dict[colors[abn_idx%len(colors)]], 2)
                cv2.drawContours(pos_visualization, contours, -1, color_dict[colors[abn_idx%len(colors)]], 2)
                cv2.drawContours(neg_visualization, contours, -1, color_dict[colors[abn_idx%len(colors)]], 2)

            visualization = np.hstack((mamm_img, pos_visualization, neg_visualization))
            visualization = PIL.Image.fromarray(visualization)

            caption_text = f'GradCAM {total_images:06d}; Label={label[idx]} (Left: Org Mamm, Center: Malignant CAM, Right: Benign CAM)'
            aim_viss.append(aim.Image(visualization, caption=caption_text))
            total_images += 1

    return aim_viss