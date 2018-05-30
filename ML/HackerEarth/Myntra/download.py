# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in JPG

# Copied off a Kaggle competition kernel. 
# However, some minor modifications to the original script have been made.

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm


def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [(line[6], line[4]) for line in csvreader if len(line[4]) > 0]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
        with open(filename, 'wb') as img_file:
            img_file.write(image_data)
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1
    
    return 0


def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=6)  # Number of CPUs to dedicate to download
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    loader()
