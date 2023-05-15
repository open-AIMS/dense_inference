import pandas as pd

from src.datagenerators import IdxDatagen_pts
from src.utils import get_ims, get_abs_path, make_run_name_df
from src.index import load_fx_model

import os
from tqdm import tqdm

import argparse


def do_classification(csv_path,
                      im_dir,
                      im_ext,
                      classifier_path,
                      batch_size,
                      im_col="camera_id",
                      fx_model_path='../data/models/feature_extractor/weights.best.hdf5'):
    print("Imdir exists?", os.path.exists(im_dir))

    df = pd.read_csv(csv_path)

    df[im_col] = df[im_col].apply(lambda cm: os.path.join(im_dir, os.path.splitext(cm)[0] + '.' + im_ext))
    ims = df[im_col].unique()

    if len(ims) == 0:
        raise Exception("Couldn't find any images, check your path and extension")

    print("Found {} images to classify".format(len(ims)))

    # fx_model_path = '../data/models/feature_extractor/weights.best.hdf5'
    fx_model_path = get_abs_path(fx_model_path)
    print("Loading fx model ...")
    fx_model = load_fx_model(fx_model_path)

    # classifier_path = '../data/models/classifier/ecorrap_community_composition-latest-230404.sav'
    classifier_path = get_abs_path(classifier_path)

    run_name = make_run_name_df(csv_path)

    out_list = []

    print("Starting classification ...")
    pbar = tqdm(ims)
    # for each image in the directory
    for im in pbar:
        pbar.set_description("Classifying {}".format(os.path.basename(im)))

        # make df for this image
        im_df = df[df[im_col] == im]

        # make the generator
        gen_obj = IdxDatagen_pts(im_df,
                                 batch_size,
                                 fx_model,
                                 classifier_path,
                                 cut_divisor=8, patch_size=256)

        # add new predictions to results list
        out_list.extend(gen_obj.make_df())

    # make a dataframe of the predictions
    out_df = pd.DataFrame(out_list)

    # create the results directory
    results_path = '../results/SAM_points'
    results_path = get_abs_path(results_path)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    csv_name = os.path.basename(os.path.splitext(csv_path)[0])

    results_path = os.path.join(results_path, csv_name)
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
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='path to CSV file containing the points to classify',
                        required=True)
    parser.add_argument('-i', '--im_dir', type=str, help='Directory containing images to classify', required=True)

    parser.add_argument('-e', '--image_extension', type=str,default='JPG', help='Image extension (e.g. JPG)', required=False)
    parser.add_argument('-m', '--classifier_model',
                        default='../data/models/classifier/ecorrap_community_composition-latest-230404.sav',
                        help='Classifier model to use (only if not using the provided model)',
                        required=False)
    parser.add_argument("-b", "--batch_size", default=16)

    args = vars(parser.parse_args())
    csv = args["data"]
    im_dir = args["im_dir"]  #'../data/images/test_ims'
    im_dir = os.path.join(os.getcwd(), im_dir)
    # print(im_dir)


    im_ext = args["image_extension"]  #'JPG'
    # points_per_im = args["n_points"]
    classifier_pth = args["classifier_model"]
    batch_size = args["batch_size"]


    do_classification(csv,
                      im_dir,
                      im_ext,
                      classifier_pth,
                      batch_size)
