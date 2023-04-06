import pandas as pd

from src.datagenerators import IdxDatagen
from src.utils import timestamp
from src.index import load_fx_model


from glob import glob
import os
from pathlib import Path
from tqdm import tqdm

import argparse

def get_ims(image_dir, ext):
    imlist = glob(os.path.join(image_dir, '*'+'.{}'.format(ext.upper()))) + \
             glob(os.path.join(image_dir, '*'+'.{}'.format(ext.lower())))

    imlist = list(dict.fromkeys(imlist))
    imlist = list(dict.fromkeys(imlist))
    return imlist

def get_abs_path(pth):
    return Path(__file__).parent / pth

def make_run_name(n_points):
    ts = timestamp()
    return "{}-points_{}".format(n_points, ts)


def do_classification(im_dir,
                      im_ext,
                      points_per_im,
                      classifier_path,
                      batch_size,
                      fx_model_path = '../data/models/feature_extractor/weights.best.hdf5'):

    # im_dir = '../data/images/test_ims'
    # im_dir = get_abs_path(im_dir)
    # im_ext = 'JPG'
    # points_per_im = 100
    # batch_size=16

    print("Imdir exists?", os.path.exists(im_dir))
    ims = get_ims(im_dir, im_ext)

    if len(ims) == 0:
        raise Exception("Couldn't find any images, check your path and extension")

    print("Found {} images to classify".format(len(ims)))

    # fx_model_path = '../data/models/feature_extractor/weights.best.hdf5'
    fx_model_path = get_abs_path(fx_model_path)
    print("Loading fx model ...")
    fx_model = load_fx_model(fx_model_path)

    # classifier_path = '../data/models/classifier/ecorrap_community_composition-latest-230404.sav'
    classifier_path = get_abs_path(classifier_path)

    run_name = make_run_name(points_per_im)



    out_list = []

    print("Starting classification ...")
    pbar = tqdm(ims)
    # for each image in the directory
    for im in pbar:

        pbar.set_description("Classifying {}".format(os.path.basename(im)))
        # make the generator
        gen_obj = IdxDatagen(im,
                             points_per_im,
                             batch_size,
                             fx_model,
                             classifier_path,
                             cut_divisor=8, patch_size=256)

        # add new predictions to results list
        out_list.extend(gen_obj.make_df())

    # make a dataframe of the predictions
    out_df = pd.DataFrame(out_list)

    # create the results directory
    results_path = '../results'
    results_path = get_abs_path(results_path)

    imdir_name = os.path.basename(os.path.dirname(ims[0]))

    results_path = os.path.join(results_path, imdir_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # save results df
    outpth = os.path.join(results_path, run_name + ".csv")
    out_df.to_csv(outpth, index=False)

    print("Saved classified points to {}".format(outpth))

if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser(
        description="Function to classify a user defined number of points from images in a directory"
    )
    parser.add_argument('-i', '--im_dir', type=str, help='Directory containing images to classify', required=True)
    parser.add_argument('-n', '--n_points', type=int,help='Number of points to classify in each image', required=True)
    parser.add_argument('-e', '--image_extension', type=str,default='JPG', help='Image extension (e.g. JPG)', required=False)
    parser.add_argument('-m', '--classifier_model',
                        default='../data/models/classifier/ecorrap_community_composition-latest-230404.sav',
                        help='Classifier model to use (only if not using the provided model)',
                        required=False)
    parser.add_argument("-b", "--batch_size", default=16)

    args = vars(parser.parse_args())

    im_dir = args["im_dir"]  #'../data/images/test_ims'
    im_dir = os.path.join(os.getcwd(), im_dir)
    # print(im_dir)


    im_ext = args["image_extension"]  #'JPG'
    points_per_im = args["n_points"]
    classifier_pth = args["classifier_model"]
    batch_size = args["batch_size"]


    do_classification(im_dir,
                      im_ext,
                      points_per_im,
                      classifier_pth,
                      batch_size)
