import io

import cv2

from .config import *
from .globals import *


def cv_detect_person(frame, person_cascade, recognizer, font_face=cv2.FONT_HERSHEY_SIMPLEX):
    persons = person_cascade.detectMultiScale(frame, 1.15, 4)
    distance = 0

    for (x, y, w, h) in persons:
        equalized_frame = cv2.equalizeHist(frame)

        face_id, distance = recognizer.predict(equalized_frame[y:y + h, x:x + w])
        if distance:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

            print('Face ID detected [%d]' % face_id)
            print('Face conf detected [%s]' % distance)

            if distance < DISTANCE_LIMIT:
                if face_id == 1:  # CAREFUL with distance.
                    face_id = u"Roman"
                elif face_id == 2:  # CAREFUL with distance.
                    face_id = u"Roman2"
                elif face_id == 4:  # CAREFUL with distance.
                    face_id = u"Arantxa"
                elif face_id == 3:  # CAREFUL with distance.
                    face_id = u"Erika"
                else:
                    face_id = "Unknown"
            else:
                face_id = "Unknown"
            cv2.putText(frame, str(face_id), (x, y - 5), font_face, 2, (255, 0, 0), 2)

    return frame, distance


def cv_recognize(images_dir=BASE_CAPTURE_DIRECTORY,
                 device_id=0,
                 is_raspbian=False,
                 is_arm=False,
                 recognizer_algorithm=LBPH_RECOGNIZER):
    training_file = os.path.join(images_dir, TRAINING_FILE)
    recognizer = cv_generate_recognizer(algorithm=recognizer_algorithm)
    if not recognizer:
        raise Exception('cv_recognize: cannot set recognition algorithm [%s]' % str(recognizer_algorithm))

    recognizer.read(training_file)
    cap = None
    camera = None
    person_cascade = None
    eye_cascade = None
    is_raspberry = False

    if is_raspbian is True and is_arm is True:
        is_raspberry = True

        try:
            import picamera

            camera = picamera.PiCamera()
        except Exception as e:
            print('CRITICAL: we are in Raspberry PI and cannot import picamera')
            print('Exception: [%s]' % e)
            sys.exit(ERROR_PICAMERA_IMPORT_ERROR_IN_RPI)

        camera.resolution = (640, 480)
        # If not RPI, these are typical locations of the Haar Cascade
        # person_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
        # person_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml')
        person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    else:
        cap = cv2.VideoCapture(device_id)
        # If RPI, these are typical locations of the Haar Cascade
        # person_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
        # person_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml')
        person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    distance = 0
    min_distance = 0
    max_distance = 0
    while True:
        if is_raspberry is True:
            image_stream = io.BytesIO()
            camera.capture(image_stream, format='jpeg')
            buffer = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)

            frame = cv2.imdecode(buffer, 1)
        else:
            ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected, distance = cv_detect_person(person_cascade=person_cascade,
                                              frame=gray_frame,
                                              recognizer=recognizer)

        if distance > max_distance:
            max_distance = distance

        if min_distance == 0:
            min_distance = distance

        if distance < min_distance:
            min_distance = distance

        cv2.imshow('Detected', detected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if DEBUG is True:
        print('MAX Distance/conf [%d]' % max_distance)
        print('MIN Distance/conf [%d]' % min_distance)

    # When everything done, release the capture
    if is_raspberry is False:
        cap.release()

    cv2.destroyAllWindows()
