import torch
import random
from fastai.vision.all import (
    DataLoaders,
    ImageDataLoaders,
    URLs,
    untar_data,
    Resize,
    default_device,
    defaults,
    Path,
    resnet18,
    vision_learner,
    accuracy,
    save_model,
    PILImage,
)
from fastai.learner import Learner
import numpy as np
from torchvision import transforms
from PIL import Image


def reproducibility(seed=42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_imagenette_loader() -> DataLoaders:
    device = get_device()
    default_device(device)
    defaults.device = device
    path = untar_data(URLs.IMAGENETTE_160)
    dls = ImageDataLoaders.from_folder(
        path, seed=42, item_tfms=Resize(128), valid_pct=0.2, device=device, bs=200
    )
    return dls


def get_cifar10_loader() -> DataLoaders:
    device = get_device()
    default_device(device)
    defaults.device = device
    path = untar_data(URLs.CIFAR)
    dls = ImageDataLoaders.from_folder(
        path, seed=42, item_tfms=Resize(32), valid_pct=0.2, device=device, bs=200
    )
    return dls


def get_stanford_cars_loader() -> DataLoaders:
    device = get_device()
    default_device(device)
    defaults.device = device
    path = untar_data(URLs.CARS)
    dls = ImageDataLoaders.from_folder(
        path, seed=42, item_tfms=Resize(128), valid_pct=0.2, device=device, bs=200
    )
    return dls


def get_learner(
    dls: DataLoaders,
    model_arch=resnet18,
    name: str = "resnet18",
    ds_name="imagenette",
    avoid_print: bool = False,
) -> Learner:
    print(f"Using device: {dls.device}")
    print(f"Model name: {name}")
    print(f"Model name: {ds_name}")

    # check if learner is saved in the current directory
    if (Path.cwd() / f"{ds_name}_{name}_learner.pth").exists():
        learn = vision_learner(dls, model_arch, metrics=accuracy)
        learn.load(f"./{ds_name}_{name}_learner.pth")
        learn.model.cuda(0)
        return learn

    learn = vision_learner(dls, model_arch, metrics=accuracy)
    learn.model.cuda(0)
    learn.fine_tune(1)
    learn.save(f"{ds_name}_{name}_learner.pth")
    save_model(f"{ds_name}_{name}_learner.pth", learn, None, with_opt=False)
    return learn


def get_img(idx: int, dls: DataLoaders, learn: Learner, use_train=False) -> Image.Image:
    if use_train:
        img = dls.train_ds[idx][0]
    else:
        img = dls.valid_ds[idx][0]
    img = PILImage.create(img)
    pred_class, pred_idx, outputs = learn.predict(img)
    prob_img = outputs[pred_idx].item()
    print(
        f"Image choosed: {idx}, Classe prevista: {pred_class}, Índice: {pred_idx}, Probabilidade: {prob_img:.4f}"
    )
    return img


def get_img_score(model: Learner, img: Image, original_idx: int = -1) -> float:
    # tfms = model.dls.valid.after_item

    model_torch = model.model
    model_torch.eval()
    model_torch.to(device=model.dls.device)
    # images = [img]

    # images_tensor = (
    #     torch.stack([tfms(im) for im in images]).to(device=model.dls.device).float()
    # )

    dl = model.dls.test_dl([img])
    images_tensor = dl.one_batch()[0]
    images_tensor = images_tensor.to(device=model.dls.device).float()

    prob_imgs_com_retangulo = []
    with torch.no_grad():
        outputs = model_torch(images_tensor)
        prob_imgs_com_retangulo = torch.softmax(outputs, dim=1)

    if original_idx == -1:
        # get the predicted index from torch
        pred_idx = torch.argmax(prob_imgs_com_retangulo, dim=1).item()
        prob_imgs_com_retangulo = prob_imgs_com_retangulo[:, pred_idx].cpu().numpy()
        return prob_imgs_com_retangulo[0], pred_idx
    else:
        prob_imgs_com_retangulo = prob_imgs_com_retangulo[:, original_idx].cpu().numpy()
        return prob_imgs_com_retangulo[0], original_idx

    # if original_idx == -1:
    # _, pred_idx, outputs = model.predict(img)
    #     return outputs[pred_idx].item(), pred_idx
    # else:
    #     _, _, outputs = model.predict(img)
    #     return outputs[original_idx].item(), original_idx
