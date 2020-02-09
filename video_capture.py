import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2 as cv
import tensorflow as tf
from core.models import Yolo
from core.dataset import transform_images
from core.utils import draw_outputs, define_basics


define_basics(flags)
flags.DEFINE_string('video', '0',
                    'Number of the webcam)')
flags.DEFINE_integer('num_classes', 8, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = Yolo(FLAGS.size, classes=FLAGS.num_classes, aux=True)

    yolo.load_weights(FLAGS.weights)
    logging.info('Weights loaded')

    with open(FLAGS.classes) as classes_f:
        class_names = [c.strip() for c in classes_f.readlines()]
    logging.info('Classes loaded: ')
    logging.info(class_names)

    t_size = 20
    t_ind = 0
    times = [0] * t_size

    try:
        vid = cv.VideoCapture(int(FLAGS.video))
    except:
        vid = cv.VideoCapture(FLAGS.video)

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        input = tf.expand_dims(input, 0)
        input = transform_images(input, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, total_num = yolo.predict(input)
        t2 = time.time()
        times[t_ind] = t2-t1
        t_ind = (t_ind + 1) % t_size

        img = draw_outputs(img, (boxes, scores, classes, total_num), class_names)
        img = cv.putText(img, "Time: {:.2f}ms".format(sum(times)/t_size*1000), (0, 30),
                          cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, lineType=cv.LINE_AA)

        cv.imshow('Video capture', img)
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
