from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from core.models import Yolo
from core.utils import load_darknet_weights

flags.DEFINE_string('input', './data/yolov3-tiny.weights', 'Path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'Path to output .tf weights')

def main(_argv):
    yolo = Yolo(classes=80, training=True)
    yolo.summary()
    logging.info('Model created')

    load_darknet_weights(yolo, FLAGS.input)
    logging.info('Weights loaded')

    img = np.random.random((1, 448, 448, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
