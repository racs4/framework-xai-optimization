import clip
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np

from mestradolib.model import ModelWrapperOpenClip


def imgprocess(img, patch_size=[16, 16], scale_factor=1):
    _transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph

    ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
    img = ResizeOp(img).convert("RGB")
    return _transform(img)

def sim_qk(q, k):
    q_cls = F.normalize(q[:1,0,:], dim=-1)
    k_patch = F.normalize(k[1:,0,:], dim=-1)

    cosine_qk = (q_cls * k_patch).sum(-1)
    cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
    cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
    cosine_qk = (cosine_qk-cosine_qk_min) / (cosine_qk_max-cosine_qk_min)
    return cosine_qk

def grad_eclip(c, qs, ks, vs, attn_outputs, map_size):
    ## gradient on last attention output
    tmp_maps = []
    for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
        grad = torch.autograd.grad(
            c,
            attn_output,
            retain_graph=True)[0]

        grad_cls = grad[:1,0,:]
        v_patch = v[1:,0,:]
        cosine_qk = sim_qk(q, k).reshape(-1)
        tmp_maps.append((grad_cls * v_patch * cosine_qk[:,None]).sum(-1))

    emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
    return emap.reshape(*map_size)

def visualize(map, raw_image, resize):
    image = np.asarray(raw_image.copy())
    map = resize(map.unsqueeze(0))[0].cpu().numpy()
    color = cv2.applyColorMap((map*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret

def resample_contour(contour, n_points):
    contour = contour.reshape(-1, 2).astype(np.float32)

    # Grant the contour is closed
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    # Distance between consecutive points
    distances = np.sqrt(
        np.sum(np.diff(contour, axis=0) ** 2, axis=1)
    )

    cumulative = np.concatenate([
        [0],
        np.cumsum(distances)
    ])

    perimeter = cumulative[-1]

    if perimeter == 0:
        raise ValueError(
            "contour has no perimeter."
        )

    target_distances = np.linspace(
        0,
        perimeter,
        n_points,
        endpoint=False
    )

    points = []
    for distance in target_distances:
        idx = np.searchsorted(
            cumulative,
            distance,
            side="right"
        ) - 1

        idx = min(
            idx,
            len(contour) - 2
        )

        segment_start = contour[idx]
        segment_end = contour[idx + 1]

        segment_length = (
            cumulative[idx + 1]
            - cumulative[idx]
        )

        if segment_length == 0:
            alpha = 0
        else:
            alpha = (distance - cumulative[idx]) / segment_length

        point = (
                segment_start + alpha * (segment_end - segment_start)
        )

        points.append(point)

    return np.asarray(points, dtype=np.float32)

def extract_polygons(gradmap, raw_image, n_var, threshold=0.01):
    n_vertices = n_var // 2

    gradmap = gradmap.detach().cpu()
    gradmap -= gradmap.min()
    gradmap /= gradmap.max()

    w, h = raw_image.size
    resize = T.Resize((h, w))

    gradmap = resize(
        gradmap.unsqueeze(0)
    )[0].numpy()

    # Create mask using threshold
    mask = (gradmap >= threshold).astype(np.uint8) * 255

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        polygon = resample_contour(
            contour,
            n_vertices
        )

        polygon = np.rint(polygon).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        polygons.append(polygon)

    return polygons

def get_grad_eclip_poligons(img, modelWrapper: ModelWrapperOpenClip, threshold, n_var):
    img_preprocessed_k = imgprocess(img).cuda().unsqueeze(0) # Lowered resolution image
    # img_preprocessed_k = img # Original resolution image
    outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = modelWrapper.clip_encode_dense(img_preprocessed_k, n=1)

    text_processed = clip.tokenize(modelWrapper.labels).cuda()
    text_embedding = modelWrapper.model.encode_text(text_processed)
    text_embedding = F.normalize(text_embedding, dim=-1)
    img_embedding = F.normalize(outputs[:, 0], dim=-1)

    cosine = (img_embedding @ text_embedding.T)[0]

    grad_emaps = []
    for i, c in enumerate(cosine):
        grad_emaps.append(grad_eclip(c, qs, ks, vs, atten_outs, map_size))

    # Generate the grad only for the first label
    tmp = grad_emaps[0].clone()
    tmp -= tmp.min()
    tmp /= tmp.max()

    polygons = extract_polygons(
        tmp,
        img,
        n_var,
        threshold
    )

    if polygons is None:
        raise ValueError("None polygons detected!")

    return polygons