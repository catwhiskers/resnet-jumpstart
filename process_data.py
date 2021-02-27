import argparse
import json
import logging
import os
import sys
import tarfile

import boto3
# import tensorflow as tf
from constants import constants
from shutil import copyfile
import shutil





def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the
    # default bucket.
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--dest-dir", type=str)
    parser.add_argument("--shards", type=int)

    return parser.parse_known_args()


def process(args): 
    dest_dir = args.dest_dir
    if os.path.exists(dest_dir): 
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    root_dir = args.root_dir
    train_val = ['training', 'validation']
    categories = list(range(0, 6))
    sub_categories = ['True', 'False']
    for t in train_val: 
        for c in categories: 
            for sc in sub_categories: 
                origin_folder = os.path.join(root_dir, t, str(c), sc)
                if os.path.exists(origin_folder): 
                    f_arr = os.listdir(origin_folder)
                    dest_folder = os.path.join(dest_dir, "{}-{}".format(c, sc))
                    if not os.path.exists(dest_folder): 
                        os.mkdir(dest_folder)
                    dest_sub_folder = os.path.join(dest_folder, t)
                    if not os.path.exists(dest_sub_folder):
                        os.mkdir(dest_sub_folder)

                    for f in f_arr[0:200]: 
                        copyfile(os.path.join(origin_folder, f), os.path.join(dest_sub_folder, f))
                    

                    
if __name__ == "__main__":
    args, unknown = _parse_args()
    process(args)
                    
