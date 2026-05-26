import time
from captum.attr import (
    Saliency,
    GuidedBackprop,
    GuidedGradCam,
    IntegratedGradients,
    LayerGradCam,
)
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import torch.nn as nn
from mestradolib.mask import mask
from mestradolib.model import get_img_score
from mestradolib.visual import add_title_to_image


def get_score_and_save_literature(
    img, threshold, attribution, learner, original_score, original_idx, img_path
):
    # create a mask from attribution with threshold
    attr_mask = np.zeros_like(attribution)
    attr_mask[attribution > threshold] = 1

    print(attr_mask.shape)
    mask_area = np.sum(attr_mask) / (attr_mask.shape[0] * attr_mask.shape[1])

    # apply mask to image
    masked_image = mask(img, attr_mask)
    masked_image_score, _ = get_img_score(learner, masked_image, original_idx)
    print(masked_image_score, original_score)
    masked_image = add_title_to_image(
        masked_image,
        f"{masked_image_score:.4f}\n{original_score:.4f}",
        font_size=9,
        font_pos=(1, 2),
    )

    # save masked image
    masked_image.save(img_path)

    inverse_mask = 1 - attr_mask
    masked_image_inverse = mask(img, inverse_mask)
    masked_image_inverse_score, _ = get_img_score(
        learner, masked_image_inverse, original_idx
    )
    masked_image_inverse = add_title_to_image(
        masked_image_inverse,
        f"{masked_image_inverse_score:.4f}\n{original_score:.4f}",
        font_size=9,
        font_pos=(1, 2),
    )

    masked_image_inverse.save(img_path.replace("masked", "masked_inverse"))

    return masked_image_score, mask_area, masked_image_inverse_score


def find_last_conv2d(module):
    last_conv = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            last_conv = submodule

    if last_conv is None:
        raise ValueError("Nenhuma camada Conv2d foi encontrada neste modelo.")

    return last_conv


def model_func_saliency(model, x):
    output = model(x)
    return output


def model_func_integrated_gradients(model, original_idx, x):
    output = model(x)
    return output[:, original_idx].flatten()


def run_literature(folder_name, i, p_name, img, learn, original_score, original_idx):
    # Checa se atributo model existe, se não, usa o próprio learn como modelo
    if hasattr(learn, "model"):
        model = learn.model
    else:
        model = learn
    model.eval()
    img_tensor = pil_to_tensor(img).unsqueeze(0).float() / 255.0
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    print(f"img_tensor.shape: {img_tensor.shape}")

    sal = Saliency(lambda x: model_func_saliency(model, x))

    initial_time = time.time()
    attribution = (
        sal.attribute(img_tensor, abs=True, target=original_idx).cpu().detach().numpy()
    )
    attribution = np.abs(attribution.squeeze()).mean(axis=0)
    final_time = time.time()

    saliency_score, saliency_area, saliency_inverse_score = (
        get_score_and_save_literature(
            img,
            0.01,
            attribution,
            learn,
            original_score,
            original_idx,
            f"{folder_name}/{i}/{p_name}/masked_saliency.png",
        )
    )
    saliency_time = final_time - initial_time

    integrated_gradients = IntegratedGradients(
        lambda x: model_func_integrated_gradients(model, original_idx, x)
    )
    initial_time = time.time()
    attribution = (
        integrated_gradients.attribute(img_tensor, target=None).cpu().detach().numpy()
    )
    attribution = np.abs(attribution.squeeze()).mean(axis=0)
    final_time = time.time()

    (
        integrated_gradients_score,
        integrated_gradients_area,
        integrated_gradients_inverse_score,
    ) = get_score_and_save_literature(
        img,
        0.01,
        attribution,
        learn,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_integrated_gradients.png",
    )
    integrated_gradients_time = final_time - initial_time

    gbp = GuidedBackprop(model)
    initial_time = time.time()
    attribution = gbp.attribute(img_tensor, target=original_idx).cpu().detach().numpy()
    attribution = np.abs(attribution.squeeze()).mean(axis=0)
    final_time = time.time()

    guided_backprop_score, guided_backprop_area, guided_backprop_inverse_score = (
        get_score_and_save_literature(
            img,
            0.01,
            attribution,
            learn,
            original_score,
            original_idx,
            f"{folder_name}/{i}/{p_name}/masked_guided_backprop.png",
        )
    )
    guided_backprop_time = final_time - initial_time

    guided_gc = GuidedGradCam(model, find_last_conv2d(model))
    initial_time = time.time()
    attribution = (
        guided_gc.attribute(img_tensor, target=original_idx).cpu().detach().numpy()
    )
    attribution = np.mean(attribution.squeeze(), axis=0)
    final_time = time.time()

    guided_gradcam_score, guided_gradcam_area, guided_gradcam_inverse_score = (
        get_score_and_save_literature(
            img,
            0.01,
            attribution,
            learn,
            original_score,
            original_idx,
            f"{folder_name}/{i}/{p_name}/masked_guided_gradcam.png",
        )
    )
    guided_gradcam_time = final_time - initial_time

    layer_gc = LayerGradCam(model, find_last_conv2d(model))
    initial_time = time.time()
    attribution = layer_gc.attribute(img_tensor, target=original_idx)
    attribution = LayerGradCam.interpolate(attribution, img_tensor.shape[2:])
    attribution = attribution.cpu().detach().numpy().squeeze(0)
    attribution = np.maximum(attribution, 0).squeeze()
    final_time = time.time()

    layer_gradcam_score, layer_gradcam_area, layer_gradcam_inverse_score = (
        get_score_and_save_literature(
            img,
            0.01,
            attribution,
            learn,
            original_score,
            original_idx,
            f"{folder_name}/{i}/{p_name}/masked_layer_gradcam.png",
        )
    )
    layer_gradcam_time = final_time - initial_time

    return {
        "saliency_score": [
            saliency_score,
            saliency_time,
            saliency_area,
            saliency_inverse_score,
            1 - saliency_area,
        ],
        "integrated_gradients_score": [
            integrated_gradients_score,
            integrated_gradients_time,
            integrated_gradients_area,
            integrated_gradients_inverse_score,
            1 - integrated_gradients_area,
        ],
        "guided_backprop_score": [
            guided_backprop_score,
            guided_backprop_time,
            guided_backprop_area,
            guided_backprop_inverse_score,
            1 - guided_backprop_area,
        ],
        "guided_gradcam_score": [
            guided_gradcam_score,
            guided_gradcam_time,
            guided_gradcam_area,
            guided_gradcam_inverse_score,
            1 - guided_gradcam_area,
        ],
        "layer_gradcam_score": [
            layer_gradcam_score,
            layer_gradcam_time,
            layer_gradcam_area,
            layer_gradcam_inverse_score,
            1 - layer_gradcam_area,
        ],
    }
