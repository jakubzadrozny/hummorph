import os
import shutil

from PIL import Image
import cv2
import numpy as np


def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))


def crop_img_to_bbox(imgs, bbox, margin=0, shift=0, resize_to=512):
    img_height, img_width = imgs[0].shape[:2]
    col_from, row_from, col_to, row_to = bbox

    height = row_to - row_from
    height_margin = int(height * margin)
    shift_val = int(height * shift)
    row_from = np.maximum(0, row_from - height_margin - shift_val)
    row_to = np.minimum(img_height, row_to + height_margin - shift_val)
    height = row_to - row_from

    width = int(0.85 * 0.5 * height)
    col_center = (col_from + col_to) / 2
    col_from = int(np.maximum(0, col_center - width))
    col_to = int(np.minimum(img_width, col_from + 2*width))
    
    if resize_to is not None:
        aspect_ratio = (col_to-col_from) / height
        target_width = int(512 * aspect_ratio)

    out_imgs = []
    for img in imgs:
        cropped_img = img[row_from:row_to, col_from:col_to]
        if resize_to is not None:
            cropped_img = cv2.resize(cropped_img, (target_width, 512), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(cropped_img)

    return out_imgs


def tile_images(images, imgs_per_row=4):
    rows = []
    row = []
    imgs_per_row = min(len(images), imgs_per_row)
    for i in range(len(images)):
        row.append(images[i])
        if len(row) == imgs_per_row:
            rows.append(np.concatenate(row, axis=1))
            row = []
    if len(rows) > 2 and len(rows[-1]) != len(rows[-2]):
        rows.pop()
    imgout = np.concatenate(rows, axis=0)
    return imgout


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None, crop_to_bbox=False, margin=0.1):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    if crop_to_bbox:
        rows, cols = np.nonzero(ray_mask.reshape(height, width))
        
        col_from = int(np.min(cols))
        col_to = int(np.max(cols) + 1)
        col_from = int(np.maximum(0, col_from - margin * (col_to - col_from)))
        col_to = int(np.minimum(width, col_to + margin * (col_to - col_from)))
        
        row_from = int(np.min(rows))
        row_to = int(np.max(rows) + 1)
        row_from = int(np.maximum(0, row_from - margin * (row_to - row_from)))
        row_to = int(np.minimum(height, row_to + margin * (row_to - row_from)))
    else:
        col_from = 0
        col_to = width
        row_from = 0
        row_to = height

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))[row_from:row_to, col_from:col_to, ...]

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))[row_from:row_to, col_from:col_to, ...]

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)[row_from:row_to, col_from:col_to, ...]

    return rgb_image, alpha_image, truth_image

     
class ImageWriter():
    def __init__(self, output_dir, exp_name):
        self.image_dir = os.path.join(output_dir, exp_name)

        print(f"The rendering is saved in {self.image_dir}")
        
        # remove image dir if it exists
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        
        os.makedirs(self.image_dir, exist_ok=True)
        self.frame_idx = -1

    def append(self, image, img_name=None):
        self.frame_idx += 1
        if img_name is None:
            img_name = f"{self.frame_idx:06d}"
        save_image(image, f'{self.image_dir}/{img_name}.png')
        return self.frame_idx, img_name

    def finalize(self):
        pass
