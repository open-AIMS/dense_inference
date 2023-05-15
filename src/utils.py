from datetime import datetime
from glob import glob

import os
from pathlib import Path

def convert_time(seconds):
    mins, sec = divmod(seconds, 60)
    hour, mins = divmod(mins, 60)
    if hour > 0:
        return "{:.0f} hour, {:.0f} minutes".format(hour, mins)
    elif mins > 5:
        return "{:.0f} minutes".format(mins)
    elif mins >= 2:
        return "{:.0f} minutes, {:.0f} seconds".format(mins, sec)
    elif mins > 0:
        return "{:.0f} minute, {:.0f} seconds".format(mins, sec)
    else:
        return "{:.2f} seconds".format(sec)


def timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


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

def make_run_name_df(df_path):
    ts = timestamp()
    return "{}_{}".format(os.path.splitext(os.path.basename(df_path))[0], ts)
