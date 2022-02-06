import numpy as np
import cv2

class Compose(object):

    def __init__(self, pipelines=list()):

        self.pipelines = pipelines

    def __call__(self, image):

        for p in self.pipelines:
            image = p(image)
        return image

class Pad(object):

    def __init__(self, pad_size):

        self.pad_size = pad_size

    def __call__(self, image):
        height, width, channel = image.shape
        zero_image = np.zeros((self.pad_size[0], self.pad_size[1], channel))
        zero_image[:height, :width, ...] = image
        return zero_image

class Resize(object):

    def __init__(self, ratio=(0.8, 1.2)):
        
        self.ratio = ratio

    def random_sample_ratio(self, image):
        
        height, width, _ = image.shape
        min_ratio, max_ratio = self.ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(width * ratio), int(height * ratio)

        return scale


    def __call__(self, image):
        scale = self.random_sample_ratio(image)
        res = cv2.resize(image, dsize=scale, interpolation=cv2.INTER_CUBIC)
        return res

class RandomCrop(object):

    def __init__(self, crop_size):

        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, image):
        crop_bbox = self.get_crop_bbox(image)
        image = self.crop(image, crop_bbox)
        return image
