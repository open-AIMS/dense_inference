from PIL import Image
import numpy as np

def cut_patch(image, patch_width, patch_height, x, y):
    dimensions = get_rect_dimensions_pixels(patch_width, patch_height, x, y)
    new_image = image.crop(dimensions)
    return new_image


def get_rect_dimensions_pixels(patchwidth, patchheight, pointx, pointy):
    return [int((pointx)-(patchwidth/2)), int((pointy)-(patchheight/2)),
            int((pointx)+(patchwidth/2)), int((pointy)+(patchheight/2))]


def crop_image(img, point, patch_size, cut_divisor):
    # img = Image.open(image_path)

    width, height = img.size
    cut_size = int(height/cut_divisor)
    # cut_height = int(height/cut_divisor)

    img = cut_patch(img, cut_size, cut_size, point[0], point[1])
    img = img.resize((patch_size, patch_size), Image.NEAREST)
    return img


def generate_random_points(img, n_points, crop_divisor=8):
    width, height = img.size
    edge = int(np.ceil(height/crop_divisor))

    x = np.random.randint(edge, width-edge, n_points)
    y = np.random.randint(edge, height-edge, n_points)

    return np.column_stack([x,y])