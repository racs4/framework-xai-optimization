from PIL import Image
import numpy as np


def normalize_attribution_map(attr):
    attr = np.asarray(attr, dtype=np.float64)
    attr = attr - np.min(attr)
    max_val = np.max(attr)
    if max_val > 0:
        attr = attr / max_val
    return attr


def topk_mask(attr, keep_ratio):
    attr = normalize_attribution_map(attr)
    h, w = attr.shape[:2]
    flat = attr.reshape(-1)
    k = max(1, int(round(flat.size * keep_ratio)))
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[idx] = True
    mask = mask.reshape(h, w)
    return mask


def deletion(image, attribution_map, keep_ratio, baseline):
    img = np.array(image.convert("RGB"), dtype=np.uint8)
    attr = np.asarray(attribution_map, dtype=np.float64)
    attr = normalize_attribution_map(attr)
    mask_keep = topk_mask(attr, keep_ratio)

    # imagem em HxWxC
    img = np.array(image, copy=True)
    img_ = img.copy()

    h, w = img_.shape[:2]
    for i in range(h):
        for j in range(w):
            if not mask_keep[i, j]:
                img_[i, j] = baseline

    return Image.fromarray(img_)


def insertion(image, attribution_map, keep_ratio, baseline):
    img = np.array(image.convert("RGB"), dtype=np.uint8)
    attr = np.asarray(attribution_map, dtype=np.float64)
    attr = normalize_attribution_map(attr)
    mask_keep = topk_mask(attr, keep_ratio)

    # imagem em HxWxC
    img = np.array(baseline, dtype=np.uint8)
    img_ = img.copy()

    original = np.array(image, copy=True)

    h, w = original.shape[:2]
    for i in range(h):
        for j in range(w):
            if mask_keep[i, j]:
                img_[i, j] = original[i, j]

    return Image.fromarray(img_)


def compute_imd(insertion_curve, deletion_curve):
    diffs = [ins - del_ for ins, del_ in zip(insertion_curve, deletion_curve)]
    return float(np.mean(diffs))


def compute_insertion_deletion_metrics(
    model_wrapper,
    image,
    original_idx,
    attribution,
    baseline=(0, 0, 0),
    step_count=100,
    percent_per_step=0.005,
):

    insertion_curve = []
    deletion_curve = []

    for step in range(1, step_count + 1):
        keep_ratio = 1.0 - (step * percent_per_step)

        batch_del = []
        batch_ins = []

        # deleção
        del_img = deletion(image, attribution, keep_ratio, baseline)
        batch_del.append(del_img)

        # inserção
        ins_img = insertion(image, attribution, keep_ratio, baseline)
        batch_ins.append(ins_img)

        # avalia o modelo
        del_score = model_wrapper.get_img_score(del_img, original_idx)
        ins_score = model_wrapper.get_img_score(ins_img, original_idx)

        deletion_curve.append(del_score)
        insertion_curve.append(ins_score)

    auc_insertion = float(np.mean(insertion_curve))
    auc_deletion = float(np.mean(deletion_curve))
    imd_value = compute_imd(insertion_curve, deletion_curve)

    return {
        "insertion_curve": insertion_curve,
        "deletion_curve": deletion_curve,
        "auc_insertion": auc_insertion,
        "auc_deletion": auc_deletion,
        "imd": imd_value,
    }
