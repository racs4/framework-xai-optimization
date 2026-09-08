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

def get_grad_eclip_img(img, modelWrapper: ModelWrapperOpenClip):
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

    h, w = img.size
    resize = T.Resize((w, h))

    # Generate the grad only for the first label
    tmp = grad_emaps[0].clone()
    tmp -= tmp.min()
    tmp /= tmp.max()
    c_ret = visualize(tmp.detach().cpu(), img, resize)
    grad_image = Image.fromarray(c_ret)

    return grad_image