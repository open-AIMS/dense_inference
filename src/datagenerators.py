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


# class for indexing from a predetermined set of points
# input is df containing points from a single image and the points to index
class IdxDatagen_pts:
    def __init__(self,
                 im_df,
                 batch_size,
                 fx_model,
                 classifier_path,
                 im_col="camera_id",
                 xcol="U",
                 ycol="V",
                 cut_divisor=8, patch_size=256):

        self.im_df = im_df
        # self.n_points = n_points
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.cut_divisor = cut_divisor

        self.im_path, self.points = self.parse_df(im_col=im_col, xcol=xcol, ycol=ycol)

        self.im = Image.open(self.im_path)
        # self.points = generate_random_points(self.im, n_points)

        self.gen = self.make_generator()
        self.point_class, self.point_desc, self.point_scores = self.classify_vectors(fx_model, classifier_path)

    #
    def parse_df(self, im_col="camera_id", xcol="U", ycol="V"):
        impth = self.im_df[im_col].unique()[0]
        x = self.im_df[xcol].apply(lambda col: int(np.round(col)))
        y = self.im_df[ycol].apply(lambda col: int(np.round(col)))
        pointset = np.column_stack([x.to_numpy(), y.to_numpy()])

        return impth, pointset

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
            out.append({"image_path": self.im_path,
                        "point_x": p[0],
                        "point_y": p[1],
                        "pred_code": self.point_class[i],
                        "pred_desc": self.point_desc[i],
                        "pred_score": self.point_scores[i]})

        return out


class IdxDatagen_pts_cents:
    def __init__(self,
                 im_df,
                 batch_size,
                 fx_model,
                 classifier_path,
                 im_col="camera_id",
                 xcol="U",
                 ycol="V",
                 cent_col="SAM_centroid",
                 vertx_col="vertex_x",
                 verty_col="vertex_y",
                 cut_divisor=8, patch_size=256):

        self.im_df = im_df
        # self.n_points = n_points
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.cut_divisor = cut_divisor

        self.im_path, self.points, self.centroid, self.vertx, self.verty = self.parse_df(im_col=im_col, xcol=xcol, ycol=ycol)

        self.im = Image.open(self.im_path)
        # self.points = generate_random_points(self.im, n_points)

        self.gen = self.make_generator()
        self.point_class, self.point_desc, self.point_scores = self.classify_vectors(fx_model, classifier_path)

    #
    def parse_df(self, im_col="camera_id", xcol="U", ycol="V", cent_col="SAM_centroid", vertx_col="vertex_x", verty_col="vertex_y"):
        impth = self.im_df[im_col].unique()[0]
        x = self.im_df[xcol].apply(lambda col: int(np.round(col)))
        y = self.im_df[ycol].apply(lambda col: int(np.round(col)))
        centroids = self.im_df[cent_col].to_numpy()
        vertx = self.im_df[vertx_col].to_numpy()
        verty = self.im_df[verty_col].to_numpy()
        pointset = np.column_stack([x.to_numpy(), y.to_numpy()])

        return impth, pointset, centroids, vertx, verty

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
            out.append({"image_path": self.im_path,
                        "SAM_centroid": self.centroid[i],
                        "vertex_x": self.vertx[i],
                        "vertex_y": self.verty[i],
                        "point_x": p[0],
                        "point_y": p[1],
                        "pred_code": self.point_class[i],
                        "pred_desc": self.point_desc[i],
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