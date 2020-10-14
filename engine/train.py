import io

import cv2

from .config import *
from .globals import *


def do_train(base_images_dir=BASE_CAPTURE_DIRECTORY):
    faces = []
    labels = []

    images_dir = base_images_dir
    TRAINING_FILE = os.path.join(images_dir, 'training.xml')

    print('Opening [%s]' % images_dir)

    dirs = os.listdir(images_dir)
    for user_dir in dirs:
        if user_dir == 'status.json' or user_dir == 'training.xml':
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

    face_count = len(faces)
    label_count = len(labels)
    if DEBUG is True:
        print("Total faces: [%d]" % face_count)
        print("Total labels: [%d]" % label_count)

    #face_recognizer= cv2.createLBPHFaceRecognizer()
    #face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.FisherFaceRecognizer_create()

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(np.array(faces), np.array(labels))

    face_recognizer.save(TRAINING_FILE)
