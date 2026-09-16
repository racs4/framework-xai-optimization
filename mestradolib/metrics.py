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
    attr = np.asarray(attribution_map, dtype=np.float64)
    mask_remove = topk_mask(attr, keep_ratio)

    img_ = np.array(image.convert("RGB"), dtype=np.uint8)
    img_[mask_remove] = baseline

    return Image.fromarray(img_)


def insertion(image, attribution_map, keep_ratio, baseline):
    attr = np.asarray(attribution_map, dtype=np.float64)
    mask_keep = topk_mask(attr, keep_ratio)

    original = np.array(image.convert("RGB"), dtype=np.uint8)
    img_ = np.full_like(original, baseline)
    img_[mask_keep] = original[mask_keep]

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
    save_files=False,
):

    insertion_curve = []
    deletion_curve = []

    del_imgs_to_save = []
    ins_imgs_to_save = []

    for step in range(1, step_count + 1):
        keep_ratio = (step * percent_per_step)

        batch_del = []
        batch_ins = []

        # deleção
        del_img = deletion(image, attribution, keep_ratio, baseline)
        # save img for debugging or visualization purposes
        if save_files:
            del_imgs_to_save.append(del_img)
        batch_del.append(del_img)

        # inserção
        ins_img = insertion(image, attribution, keep_ratio, baseline)
        # save img for debugging or visualization purposes
        if save_files:
            ins_imgs_to_save.append(ins_img)
        batch_ins.append(ins_img)

        # avalia o modelo
        del_score, _ = model_wrapper.get_img_score(del_img, original_idx)
        ins_score, _ = model_wrapper.get_img_score(ins_img, original_idx)

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
        "del_imgs_to_save": del_imgs_to_save,
        "ins_imgs_to_save": ins_imgs_to_save,
    }
