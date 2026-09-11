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
    resize_images,
)
from fastai.learner import Learner
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


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
    print(f"Classes: {dls.vocab}")
    print(f"Test size: {len(dls.valid_ds)}")
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
    path_256 = path / "cars-256"
    if not path_256.exists():
        resize_images(path=path, dest=path_256, max_size=256, recurse=True)
    dls = ImageDataLoaders.from_folder(
        path_256, seed=42, item_tfms=Resize(128), valid_pct=0.2, device=device, bs=200
    )
    return dls


def get_imagewoof_loader() -> DataLoaders:
    device = get_device()
    default_device(device)
    defaults.device = device
    path = untar_data(URLs.IMAGEWOOF_160)
    dls = ImageDataLoaders.from_folder(
        path, seed=42, item_tfms=Resize(128), valid_pct=0.2, device=device, bs=200
    )
    return dls


def get_flowers_loader() -> DataLoaders:
    device = get_device()
    default_device(device)
    defaults.device = device
    path = untar_data(URLs.FLOWERS)
    path_256 = path / "flowers-256"
    if not path_256.exists():
        resize_images(path=path, dest=path_256, max_size=256, recurse=True)
    dls = ImageDataLoaders.from_folder(
        path_256, seed=42, item_tfms=Resize(128), valid_pct=0.2, device=device, bs=200
    )
    return dls


def get_food_loader() -> DataLoaders:
    device = get_device()
    default_device(device)
    defaults.device = device
    path = untar_data(URLs.FOOD)
    path_256 = path / "food-256"
    if not path_256.exists():
        resize_images(path=path, dest=path_256, max_size=256, recurse=True)
    dls = ImageDataLoaders.from_folder(
        path_256, seed=42, item_tfms=Resize(128), valid_pct=0.2, device=device, bs=200
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


def get_img(idx: int, dls: DataLoaders, use_train=False) -> Image.Image:
    if use_train:
        img = dls.train_ds[idx][0]
    else:
        img = dls.valid_ds[idx][0]
    img = PILImage.create(img)
    print(f"Image choosed: {idx}, Image size: {img.size}")
    # pred_class, pred_idx, outputs = learn.predict(img)
    # prob_img = outputs[pred_idx].item()
    # print(
    #     f"Image choosed: {idx}, Classe prevista: {pred_class}, Índice: {pred_idx}, Probabilidade: {prob_img:.4f}"
    # )
    return img


def get_img_score(model: Learner, img: Image, original_idx: int = -1):
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
        prob_imgs_com_retangulo = prob_imgs_com_retangulo[:, pred_idx].cpu(
        ).numpy()
        return prob_imgs_com_retangulo[0], pred_idx
    else:
        prob_imgs_com_retangulo = prob_imgs_com_retangulo[:, original_idx].cpu(
        ).numpy()
        return prob_imgs_com_retangulo[0], original_idx

    # if original_idx == -1:
    # _, pred_idx, outputs = model.predict(img)
    #     return outputs[pred_idx].item(), pred_idx
    # else:
    #     _, _, outputs = model.predict(img)
    #     return outputs[original_idx].item(), original_idx


class ModelWrapper:
    def __init__(self):
        self.run_literature = True
        pass

    def get_vetorized_probabilities(self, imgs, pred_idx_original):
        pass

    def get_img_score(self, img: Image, original_idx: int = -1):
        pass

    def get_pytorch_model(self):
        pass


class ModelWrapperFastAI(ModelWrapper):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_vetorized_probabilities(self, imgs, pred_idx_original):
        model_torch = self.model.model
        model_torch.eval()
        model_torch.to(device=self.model.dls.device)

        dl = self.model.dls.test_dl(imgs)
        images_tensor = dl.one_batch()[0]
        images_tensor = images_tensor.to(device=self.model.dls.device).float()

        with torch.no_grad():
            outputs = self.model.model(images_tensor)
            prob_imgs_com_retangulo = torch.softmax(outputs, dim=1)

        prob_imgs_com_retangulo = (
            prob_imgs_com_retangulo[:, pred_idx_original].cpu().numpy()
        )

        return prob_imgs_com_retangulo

    def get_img_score(self, img: Image, original_idx: int = -1):
        return get_img_score(self.model, img, original_idx)

    def get_pytorch_model(self):
        return self.model.model


class ModelWrapperClip(ModelWrapper):
    def __init__(self, labels):
        super().__init__()
        self.run_literature = False
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.labels = labels

    def get_vetorized_probabilities(self, imgs, pred_idx_original):
        inputs = self.processor(
            text=self.labels,
            images=imgs,
            return_tensors="pt",
            padding=True,
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs[:, pred_idx_original].cpu().detach().numpy()

        return probs

    def get_img_score(self, img: Image, original_idx: int = -1):
        inputs = self.processor(
            text=self.labels,
            images=[img],
            return_tensors="pt",
            padding=True,
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        if original_idx == -1:
            result = probs.argmax().item()
            return probs[0, result].item(), result
        else:
            return probs[0, original_idx].item(), original_idx

    def get_pytorch_model(self):
        return self.model
