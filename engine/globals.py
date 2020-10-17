import os
import sys
import json
import getopt

import cv2
import numpy as np
import fnmatch
from pathlib import Path

from .config import *


def show_help(program_name):
    print('''
    Usage:
    %s:
        -u <user> || --user=<user> (Note this is required for capture)

        -d <device_id> || --device_id=<device_id>
           (if not in Raspberry PI you must pass a device id as in
           the camera device located at /dev/video[id])

        -o <output_directory> || --output=<output_directory>
           (where images will be stored, the format will be the
           directory passed with the username joined, for example
           "output/username/". There you will find several images:
           - <number>.jpg: the captured image.
           - <number>_boxed.jpg: the captured image with a red box
           around the detected face/s.
           - <number>_gray.jpg: the captured image in gray scale.

           Capture, Train and Identify requires -o or will default to
           BASE_CAPTURE_DIRECTORY if not passed in an explicit way.

        -l <limit> || --limit=<limit>
           (a limit for the number of images to save. By default 0
           that means is unlimited).
           
        -a <algorithm number> || --algorithm=<algorithm number>
           (the algorithm can be 1 to 3, corresponding to LBPH_RECOGNIZER, 
           FISHER_RECOGNIZER and EIGEN_RECOGNIZER).
           
        -s || --silent
           (do not show the positive image on screen)

        -c || --capture
          (capture images to be trained. Remember to set device id if not in
          a Raspberry PI, -d 1 or so)

        -t || --train
          (train models from images in output_dir/images_dir and generate XML
          file with the trained model)

        -r || --recognize
          (identify people opening the camera -do not forget the device id if
          not in a Raspberry PI- and matching against the trained model)

        -n || --no-extra-images
          (this flag will limit the image generation just to what is useful
          to train the model, not creating color, grey, additional images).

        -h || --help   
    ''' % program_name)


def parse_arguments(argv, program_name='application'):
    parsed_dict = dict()
    parsed_dict['user'] = None
    parsed_dict['did'] = 0
    parsed_dict['silent'] = False
    parsed_dict['output_dir'] = BASE_CAPTURE_DIRECTORY
    parsed_dict['limit'] = 10
    parsed_dict['algorithm'] = LBPH_RECOGNIZER
    parsed_dict['create_extra_images'] = True
    parsed_dict['wanna_capture'] = False
    parsed_dict['wanna_train'] = False
    parsed_dict['wanna_recognize'] = False

    try:
        opts, args = getopt.getopt(argv,
                                   "ctrshnu:d:o:l:a:",
                                   [
                                       "capture",
                                       "train",
                                       "recognize",
                                       "silent",
                                       "help",
                                       "no-extra-images",
                                       "user=",
                                       "device_id=",
                                       "output=",
                                       "limit=",
                                       "algorithm="
                                   ])
    except getopt.GetoptError as e:
        print('* Error in arguments: [%s]' % str(e))
        show_help(program_name=program_name)
        sys.exit(ERROR_PARSING_ARGUMENTS)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            show_help(program_name=program_name)
            sys.exit(ERROR_OK)
        elif opt in ("-c", "--capture"):
            parsed_dict['wanna_capture'] = True
        elif opt in ("-s", "--silent"):
            parsed_dict['silent'] = True
        elif opt in ("-n", "--no-extra-images"):
            parsed_dict['create_extra_images'] = False
        elif opt in ("-u", "--user"):
            parsed_dict['user'] = arg
        elif opt in ("-d", "--device_id"):
            parsed_dict['did'] = arg
            try:
                parsed_dict['did'] = int(parsed_dict['did'])
            except Exception as e:
                print('* Device ID passed was invalid [%s]' % str(parsed_dict['did']))
                show_help(program_name=program_name)
                sys.exit(ERROR_DEVICE_ID_IS_NOT_INTEGER)
        elif opt in ("-o", "--output"):
            parsed_dict['output_dir'] = arg
        elif opt in ("-l", "--limit"):
            parsed_dict['limit'] = arg
            try:
                parsed_dict['limit'] = int(parsed_dict['limit'])
            except Exception as e:
                print('* Limit passed was invalid [%s]' % str(parsed_dict['limit']))
                show_help(program_name=program_name)
                sys.exit(ERROR_LIMIT_IS_NOT_INTEGER)
        elif opt in ("-a", "--algorithm"):
            parsed_dict['algorithm'] = arg
            try:
                parsed_dict['algorithm'] = int(parsed_dict['algorithm'])
            except Exception as e:
                print('* Algorithm passed was invalid [%s]' % str(parsed_dict['algorithm']))
                show_help(program_name=program_name)
                sys.exit(ERROR_ALGORITHM_IS_NOT_INTEGER)

            if 3 < parsed_dict['algorithm'] < 1:
                print('* Algorithm passed was invalid [%s] (allowed 1, 2 or 3).' % str(parsed_dict['algorithm']))
                show_help(program_name=program_name)
                sys.exit(ERROR_ALGORITHM_VALUE_NOT_VALID)

            if parsed_dict['algorithm'] == 2:
                print('* Algorithm FISHER/LDA requires MORE THAN ONE DIFFERENT LABEL. Please, capture for at least two users')
        elif opt in ("-t", "--train"):
            parsed_dict['wanna_train'] = True
        elif opt in ("-r", "--recognize"):
            parsed_dict['wanna_recognize'] = True

    if parsed_dict['wanna_capture'] is True and not parsed_dict['user']:
        print('* User is required when capturing')
        show_help(program_name=program_name)
        sys.exit(ERROR_USER_IS_REQUIRED_WHEN_CAPTURING)

    if parsed_dict['did'] == 0:
        print('** IMPORTANT: set the default capture device id to 0: /dev/video0. This may fail if the video capture device is not this one.')

    return parsed_dict


def cv_set_recognition_algorithm_treshold(algorithm=LBPH_RECOGNIZER):
    POSITIVE_THRESHOLD_VALUE = 3000

    if algorithm < 1 or algorithm > 3:
        print("WARNING: face algorithm must be in the range 1-3")
        sys.exit(ERROR_INVALID_RECOGNITION_ALGORITHM)

    # Threshold for the confidence of a recognized face before it's
    # considered a positive match.  Confidence values below this
    # threshold will be considered a positive match because the lower
    # the confidence value, or distance, the more confident the
    # algorithm is that the face was correctly detected.  Start with a
    # value of 3000, but you might need to tweak this value down if
    # you're getting too many false positives (incorrectly recognized
    # faces), or up if too many false negatives (undetected faces).
    # POSITIVE_THRESHOLD = 3500.0
    if algorithm == 1:
        POSITIVE_THRESHOLD_VALUE = 80
    elif algorithm == 2:
        POSITIVE_THRESHOLD_VALUE = 250
    else:
        POSITIVE_THRESHOLD_VALUE = 3000

    return POSITIVE_THRESHOLD_VALUE


def cv_crop(image, x, y, w, h):
    """
    We pass an opencv image (image that can be understood as Numpy array).

    We pass x, y, w and h as the dimensions variables. IMPORTANT do remember
    that in Numpy/Image y is first (the upper left corner).

    The resulting image cropped to CROP_WIDTH/CROP_HEIGHT will be returned, as
    an additional pass it will dimensioned to the specific dimensions that are
    better for training.
    """
    crop_height = int((CROP_HEIGHT / float(CROP_WIDTH)) * w)
    mid_y = y + h / 2
    y1 = int(max(0, mid_y - crop_height / 2))
    y2 = int(min(image.shape[0] - 1, mid_y + crop_height / 2))
    x = int(x)
    w = int(w)

    cropped_image = image[y1:y2, x:x + w]

    return cv2.resize(cropped_image, (CROP_WIDTH, CROP_HEIGHT),
                      interpolation=cv2.INTER_LANCZOS4)


def iterate_image_files(directory, match='*_crop.jpg'):
    """Generator function to iterate through all files in a directory recursively
    which match the given filename match parameter: *_crop.jpg
    """
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, match):
            yield os.path.join(root, filename)


def cv_normalize_array(data_array, low, high, dtype=None):
    """
    Normalizes a given array in X to a value between low and high.
    Adapted from python OpenCV face recognition example at:
    https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py
    """
    data_array = np.asarray(data_array)
    minX, maxX = np.min(data_array), np.max(data_array)

    # normalize to [0...1].
    data_array = data_array - float(minX)
    data_array = data_array / float((maxX - minX))

    # scale to [low...high].
    data_array = data_array * (high - low)
    data_array = data_array + low

    if dtype is None:
        return np.asarray(data_array)

    return np.asarray(data_array, dtype=dtype)


def open_images_status_file(images_status_file=IMAGES_STATUS_FILE, base_images_dir=BASE_CAPTURE_DIRECTORY):
    initial_content = ['{"user_count": 0}']
    image_status_file = os.path.join(base_images_dir, images_status_file)
    fp = None
    status = None

    if os.path.exists(image_status_file):
        try:
            fp = open(image_status_file, 'rt')
        except Exception as e:
            print('open_images_status_file: exception reading file -> exception [%s]' % e)
            sys.exit(ERROR_READING_STATUS_FILE)
    else:
        # Create if directory does not exist, ignore either.
        Path(base_images_dir).mkdir(parents=True, exist_ok=True)

        fp = open(image_status_file, 'wt')
        fp.writelines(initial_content)
        fp.close()

        return json.loads('{"user_count": 0}')

    with fp:
        status = json.load(fp)
        return status


def open_user_id_file(user, base_images_dir=BASE_CAPTURE_DIRECTORY, last_user_id=0):
    user_id_dir = os.path.join(base_images_dir, user)
    user_id_file = os.path.join(user_id_dir, 'id')

    fp = None
    status = None

    if os.path.exists(user_id_file):
        try:
            fp = open(user_id_file, 'rt')
        except Exception as e:
            print('open_user_id_file: exception reading file -> exception [%s]' % e)
            sys.exit(ERROR_READING_USER_ID_FILE)
    else:
        user_id = last_user_id + 1
        # Create if directory does not exist, ignore either.
        if DEBUG is True:
            print('=====> Mkdir [%s]' % user_id_dir)
        Path(user_id_dir).mkdir(parents=True, exist_ok=True)

        fp = open(user_id_file, 'wt')
        fp.write(str(user_id))
        fp.close()

        return user_id

    with fp:
        user_id = fp.read()

    try:
        user_id = int(user_id)
    except Exception as e:
        print('open_user_id_file: exception in user_id. Maybe not integer -> exception [%s]' % e)
        sys.exit(ERROR_USER_ID_IS_NOT_INTEGER)

    if user_id == 0:
        print('open_user_id_file: user_id cannot be 0.')
        return 0

    return user_id


def update_status(status, images_status_file=IMAGES_STATUS_FILE, base_images_dir=BASE_CAPTURE_DIRECTORY):
    image_status_file = os.path.join(base_images_dir, images_status_file)
    with open(image_status_file, 'wt') as fp:
        fp.write(json.dumps(status))


def cv_generate_recognizer(algorithm, threshold=cv2.THRESH_OTSU):
    if algorithm == 1:
        return cv2.face.LBPHFaceRecognizer_create(threshold=threshold)
    elif algorithm == 2:
        return cv2.face.FisherFaceRecognizer_create(threshold=threshold)
    elif algorithm == 3:
        return cv2.face.EigenFaceRecognizer_create(threshold=threshold)
    else:
        print("WARNING: face algorithm must be LBPH, Fisher or Eigen (1-3). See config.py.")
        return None
