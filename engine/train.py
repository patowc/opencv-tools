import json

import cv2

from .config import *
from .globals import *


def do_train(base_images_dir=BASE_CAPTURE_DIRECTORY, recognizer_algorithm=LBPH_RECOGNIZER):
    faces = []
    labels = []
    detectables_dict = {'unknown': 0}

    images_dir = base_images_dir
    TRAINING_FILE = os.path.join(images_dir, 'training.xml')

    print('Opening [%s]' % images_dir)

    dirs = os.listdir(images_dir)
    for user_dir in dirs:
        if user_dir == 'status.json' or user_dir == 'training.xml' or user_dir == 'detectables.json':
            continue

        user_dir_full_path = os.path.join(images_dir, user_dir)
        if DEBUG is True:
            print('=====> open_user_id_file: user=[%s] base:dir=[%s]' % (user_dir,
                                                                         base_images_dir))
        user_id = open_user_id_file(user_dir, base_images_dir=base_images_dir)

        for filename in iterate_image_files(directory=user_dir_full_path):
            print('FILENAME: %s' % filename)
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces.append(image)
            labels.append(user_id)

            detectables_dict[user_dir] = user_id

    face_count = len(faces)
    label_count = len(labels)
    if DEBUG is True:
        print("Total faces: [%d]" % face_count)
        print("Total labels: [%d]" % label_count)

    face_recognizer = cv_generate_recognizer(algorithm=recognizer_algorithm)
    if not face_recognizer:
        raise Exception('do_train: invalid recognizer algorithm [%s]' % str(recognizer_algorithm))
    face_recognizer.train(np.array(faces), np.array(labels))

    face_recognizer.save(TRAINING_FILE)

    generate_detectables_json(detectables=detectables_dict)
