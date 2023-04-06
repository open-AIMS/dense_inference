Code to use a reefcloud trained model to inference a user defined number of points on a set of images


### Installation
create and activate a virtual/conda environment, then inside the root directory of the project run:

    pip install -e .
    pip install -r requirements.txt

### Usage
To classify some images, make a new directory inside `data/images/` and add in your images. They should all have the same extension

From the root directory of the project run:

    python scripts/inference_image_dir.py -i 'path/to/image_directory' -n n_points -e 'JPG'

For example, if the images are JPGs and are in `data/images/test_ims`

    python scripts/inference_image_dir.py -i 'data/images/test_ims' -n 5 -e 'JPG'

would classify 5 random points in each image and save the results in `results/test_ims/`


