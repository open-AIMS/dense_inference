import pandas as pd
from tensorflow.keras.utils import Sequence, img_to_array
import numpy as np
from PIL import Image


from src.img_utils import generate_random_points, crop_image
from src.index import index
from src.classify import classify


"""
Process:
- Load image
- generate n random points
- create generator from image and the points
"""
class IdxDatagen:
    def __init__(self, im_path, n_points, batch_size, fx_model, classifier_path, cut_divisor=8, patch_size=256):
        self.im_path = im_path
        # self.n_points = n_points
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.cut_divisor= cut_divisor

        self.im = Image.open(im_path)
        self.points = generate_random_points(self.im, n_points)

        self.gen = self.make_generator()
        self.point_class, self.point_desc, self.point_scores = self.classify_vectors(fx_model, classifier_path)

    def cropping_fn(self, im, point):
        patch = crop_image(im, point, self.patch_size, self.cut_divisor)
        patch = img_to_array(patch)
        return patch

    def make_generator(self):
        gen = RandomPointCroppingLoader(self.im, self.points, self.batch_size, self.cropping_fn)
        return gen

    def classify_vectors(self, fx_model_pth, classifier_pth):
        vectors = index(self.gen, fx_model_pth)
        return classify(vectors, classifier_pth)

    def make_df(self):
        out = []
        for i, p in enumerate(self.points):
            out.append({"image_path" : self.im_path,
                        "point_x" : p[0],
                        "point_y" : p[1],
                        "pred_code" : self.point_class[i],
                        "pred_desc" : self.point_desc[i],
                        "pred_score": self.point_scores[i]})

        return out


class RandomPointCroppingLoader(Sequence):

    def __init__(self, im, points, batch_size, cropping_fn):
        self.im = im
        self.points = points
        self.batch_size = batch_size
        self.cropping_fn = cropping_fn

    def __len__(self):
        return int(np.ceil(len(self.points) / float(self.batch_size)))

    def __getitem__(self, idx):
        batchpoints = self.points[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            self.cropping_fn(self.im, batchpoints[i]) for i in range(len(batchpoints))
        ])



if __name__ == "__main__":
    impth = '../data/images/test_ims/202203_ONLI_BA1_S_P1_EC1_4760.JPG'
    dg = IdxDatagen(impth, 10, 5)
    dg.gen.__getitem__(0)