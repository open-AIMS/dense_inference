Code to use a reefcloud trained model to inference a user defined number of points on a set of images


### Installation
create and activate a virtual/conda environment, then inside the root directory of the project run:

    pip install -e .
    pip install -r requirements.txt

### Usage
1. To classify some images, make a new directory inside `data/images/` and add in your images. They should all have the same extension

From the root directory of the project run:

    python scripts/inference_image_dir.py -i 'path/to/image_directory' -n n_points -e 'JPG'

For example, if the images are JPGs and are in `data/images/test_ims`

    python scripts/inference_image_dir.py -i 'data/images/test_ims' -n 5 -e 'JPG'

would classify 5 random points in each image and save the results in `results/test_ims/`

2. To classify points from a csv (e.g. from the SAM outputs)

From the root directory of the project run:

    scripts/inference_from_csv.py -d 'path/to/points.csv' -i 'path/to/image_directory' -e 'JPG'

For example, if the csv is `data/images/SAM_points_test.csv` and images are JPGs and are in `data/images/test_ims`

    scripts/inference_from_csv.py -d 'data/images/SAM_points_test.csv' -i data/images/test_ims/ -e 'JPG'

would classify the csv points and save the results in `results/SAM_points/`



