"""cv_detect_train.pu
Multiplatform OpenCV (cv2) face detection and capture tool.
The MIT License (MIT)

Copyright (c) 2020 Román Ramírez Giménez (MIT License)

Run this script to detect faces, save images and then be able to train models
for different tools (for example, Magic Mirror 2 facial recognition ones).

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
           that means is unlimitted).

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
"""
import os
import sys

from engine.config import *
from engine.globals import *
from engine.capture import *
from engine.train import *
from engine.recognize import *


if __name__ == "__main__":
    IS_RASPBIAN = False
    IS_ARM = False

    sysname, nodename, release, version, machine = os.uname()
    nodename = nodename.lower()
    if DEBUG is True:
        print('OS: %s' % os.name)
        print('Uname: sys[%s] node[%s] rel[%s] ver[%s] machine[%s]' % (sysname,
                                                                       nodename,
                                                                       release,
                                                                       version,
                                                                       machine))

    if nodename == 'raspberrypi' or nodename == 'raspbian':
        IS_RASPBIAN = True
        if DEBUG is True:
            print('==> IS raspbian environment')
    else:
        if DEBUG is True:
            print('==> Not in raspbian environment')

    if 'arm' in machine.lower():
        IS_ARM = True
        if DEBUG is True:
            print('==> IS ARM architecture')
    else:
        if DEBUG is True:
            print('==> Not in ARM architecture')

    if len(sys.argv) < 2:
        print('*: Please, check arguments passed')
        show_help(program_name=sys.argv[0])
        sys.exit(ERROR_INVALID_ARGUMENT_COUNT)

    parsed_dict = parse_arguments(argv=sys.argv[1:],
                                  program_name=sys.argv[0])

    images_status = open_images_status_file(base_images_dir=parsed_dict['output_dir'])
    if not images_status:
        print('CRITICAL: something critical happened reading images status file. EXIT.')
        sys.exit(-1)

    if parsed_dict['wanna_train'] is True:
        do_train(base_images_dir=parsed_dict['output_dir'],
                 recognizer_algorithm=parsed_dict['algorithm'])
    elif parsed_dict['wanna_recognize'] is True:
        cv_recognize(images_dir=parsed_dict['output_dir'],
                     device_id=parsed_dict['did'],
                     recognizer_algorithm=parsed_dict['algorithm'],
                     is_raspbian=IS_RASPBIAN, is_arm=IS_ARM)
    elif parsed_dict['wanna_capture'] is True:
        do_capture(user=parsed_dict['user'],
                   device_id=parsed_dict['did'],
                   is_raspbian=IS_RASPBIAN,
                   is_arm=IS_ARM,
                   silent_mode=parsed_dict['silent'],
                   base_images_dir=parsed_dict['output_dir'],
                   limit=parsed_dict['limit'],
                   status=images_status,
                   create_extra_images=parsed_dict['create_extra_images'])
    else:
        show_help(program_name=sys.argv[0])
        sys.exit(ERROR_NO_ACTION_SELECTED)
