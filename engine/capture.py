import io

from pathlib import Path
import cv2

from .config import *
from .globals import *


def do_capture(user,
               device_id=0,
               is_raspbian=False,
               is_arm=False,
               silent_mode=False,
               base_images_dir=BASE_CAPTURE_DIRECTORY,
               limit=0,
               status=None,
               create_extra_images=True):
    is_raspberry = False
    cap = None
    camera = None
    face_cascade = None
    last_user_id = 0

    images_dir = os.path.join(base_images_dir, user)

    Path(images_dir).mkdir(parents=True, exist_ok=True)
    if status:
        last_user_id = status['user_count']
    else:
        last_user_id = 0

    if DEBUG is True:
        print('=====> open_user_id_file: user=[%s] base:dir=[%s] last_id=[%s]' % (user,
                                                                                  base_images_dir,
                                                                                  str(last_user_id)))

    user_id = open_user_id_file(user=user, base_images_dir=base_images_dir, last_user_id=last_user_id)
    if user_id == 0:
        sys.exit(5)
    else:
        print('=> Update status to new user_id=[%d]' % user_id)
        status['user_count'] = user_id
        update_status(status=status, base_images_dir=base_images_dir)

    if is_raspbian is True and is_arm is True:
        is_raspberry = True

        try:
            import picamera

            camera = picamera.PiCamera()
        except Exception as e:
            print('CRITICAL: we are in Raspberry PI and cannot import picamera')
            print('Exception: [%s]' % e)
            sys.exit(-1)

        camera.resolution = (640, 480)
        face_cascade = cv2.CascadeClassifier(
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
    else:
        cap = cv2.VideoCapture(device_id)
        face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml')

    image_save_count = 0
    while True:
        ret = None
        frame = None

        # Capture frame-by-frame
        if is_raspberry is True:
            image_stream = io.BytesIO()
            camera.capture(image_stream, format='jpeg')
            buffer = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)

            frame = cv2.imdecode(buffer, 1)
        else:
            ret, frame = cap.read()

        # Frame now is normalized, no matter if rpi or not.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.2,
                                              minNeighbors=5,
                                              minSize=(30, 30))

        faces_found = False
        face_counter = 0
        original_frame = frame.copy()
        cropped_frame = None
        for (x, y, w, h) in faces:
            faces_found = True
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 0, 255),
                          5)

            cropped_frame = cv_crop(gray, x, y, w, h)

            face_counter += 1

        # We can show gray instead of color.
        # cv2.imshow('frame',gray)
        if faces_found is True:
            if silent_mode is True:
                print('Found a FACE [counter=%d]' % face_counter)
            else:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if face_counter > 1:
                print('** IMPORTANT: more than one face was found. This is not good for training!')

            # We will make several copies for the user to be able to verify
            # every step of the capture process.
            #
            # The most relevant are *_crop.jpg, that are images ready for the
            # training process.
            cropped_image_name = '%s_crop.jpg' % str(image_save_count)  # Trainee

            if create_extra_images is True:
                image_name = '%s.jpg' % str(image_save_count)
                boxed_image_name = '%s_boxed.jpg' % str(image_save_count)
                gray_image_name = '%s_gray.jpg' % str(image_save_count)

                cv2.imwrite(os.path.join(images_dir, image_name), original_frame)
                cv2.imwrite(os.path.join(images_dir, boxed_image_name), frame)
                cv2.imwrite(os.path.join(images_dir, gray_image_name), gray)

            # Always must create these ones
            cv2.imwrite(os.path.join(images_dir, cropped_image_name), cropped_frame)  # Trainee

            image_save_count += 1

            if limit > 0:
                if image_save_count == limit:
                    print('===> Image limit reached [%s]. Exit.' % image_save_count)

                    if is_raspberry is False:
                        cap.release()

                    cv2.destroyAllWindows()
                    break

    # When everything done, release the capture
    if is_raspberry is False:
        cap.release()

    cv2.destroyAllWindows()

