from PIL import Image, ImageDraw
import numpy as np


def mask(img, mask, color="black"):
    new_img = img.copy()

    # reshape mask to be 2D instead of 1D
    if len(mask.shape) == 1:
        mask = mask.reshape(img.size[1], img.size[0])

    mask = (mask > 0).astype(np.uint8) * 255
    mask = Image.fromarray(mask).convert("L")
    new_img.paste(color, mask=mask)
    return new_img


def mask_rectangle(img, box, color="black"):
    new_img = img.copy()
    new_img.paste(color, box)
    return new_img


def mask_inverse_rectangle(img, box, color="black"):
    new_img = img.copy()
    mask = Image.new("L", img.size, 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=0)
    new_img.paste(color, mask=mask)
    return new_img


def mask_polygon(img, points, color="black"):
    polygon_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)
    draw.polygon(polygon_points, fill=color)

    return new_img


def dynamic_mask_polygon(img, arr, color="black"):
    n = arr[0]
    points = arr[1 : (n * 2 + 1)]
    return mask_polygon(img, points, color)


def positive_mask_polygon(img, points):
    # Create a blank heatmap
    final_mask = np.zeros((img.height, img.width), dtype=np.int32)

    polygon_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

    mask = Image.new("L", (img.width, img.height), 0)

    ImageDraw.Draw(mask).polygon(polygon_points, outline=1, fill=1)

    final_mask += np.array(mask)

    img_copy = img.copy()
    img_array = np.array(img_copy)

    # Create a mask where heatmap_solutions is greater than 0
    mask = final_mask > 0

    # Set the pixels in the original image to black where the mask is False
    img_array[~mask] = [0, 0, 0]

    # Convert back to PIL Image
    return Image.fromarray(img_array)
