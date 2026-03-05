import cv2
import numpy as np
from PIL import Image, ImageDraw


def inpaint_image_with_rectangle(img, rectangle):
    # Create a mask for the area to inpaint
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    np_img = np.array(img)
    x1, y1, x2, y2 = rectangle
    mask[y1:y2, x1:x2] = 255

    # Inpaint the image using the mask
    inpainted = cv2.inpaint(np_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def inpaint_inverse_image_with_rectangle(img, rectangle):
    # Create a mask for the area to inpaint
    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    np_img = np.array(img)
    x1, y1, x2, y2 = rectangle
    mask[y1:y2, x1:x2] = 0

    # Inpaint the image using the mask
    inpainted = cv2.inpaint(np_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def inpaint_image_with_polygon(img, polygon_points):
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=255)
    mask = np.array(mask)
    np_img = np.array(img)

    inpainted = cv2.inpaint(np_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def inpaint_inverse_image_with_polygon(img, polygon_points):
    mask = Image.new("L", img.size, 255)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=0)
    mask = np.array(mask)
    np_img = np.array(img)

    inpainted = cv2.inpaint(np_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def inpaint_image_with_points(img, points):
    polygon_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    return inpaint_image_with_polygon(img, polygon_points)


def inpaint_inverse_image_with_points(img, points):
    polygon_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    return inpaint_inverse_image_with_polygon(img, polygon_points)


def dynamic_inpaint_image_with_points(img, arr):
    n = arr[0]
    points = arr[1 : (n * 2 + 1)]
    return inpaint_image_with_points(img, points)


def inpaint_image_with_mask(img, mask):
    np_img = np.array(img)
    np_mask = np.array(mask).astype(np.uint8)
    inpainted = cv2.inpaint(np_img, np_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def save_inpaint_gif_result(img, result, title):
    images = []
    for [x1, y1, x2, y2] in result.X:
        inpainted_image = inpaint_image_with_rectangle(img, (x1, y1, x2, y2))
        images.append(inpainted_image)
    images[0].save(
        f"{title}.gif", save_all=True, append_images=images[1:], duration=500, loop=0
    )
