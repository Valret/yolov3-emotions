import sys
sys.path.append('../yolov3-diploma')

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import traceback
from core.utils import get_example
from absl import app, flags
from absl.flags import FLAGS


flags.DEFINE_string('input', './data/ck+/emotions_dataset', 'Path to initial dataset')
flags.DEFINE_string('output', 'datasets/ck_dataset_flip.tfrecord', 'Path to preprocessed dataset')

# CK+ version:
class_map = {0:'Anger', 1:'Contempt', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Sadness', 6:'Surprise', 7:'Neutral'}
# Microsoft version:
map_to = {'Anger':0, 'Contempt':1, 'Disgust':2, 'Fear':3, 'Happy':4, 'Neutral':5, 'Sadness':6, 'Surprise':7}

def main(args):
    detector = cv.dnn.readNetFromCaffe('model_weights/deploy.prototxt',
                                       'model_weights/res10_300x300_ssd_iter_140000_fp16.caffemodel')
    writer = tf.io.TFRecordWriter(FLAGS.output)
    ind = 0

    try:
        for dirname, dirs, file in os.walk(FLAGS.input):
            if '.DS_Store' in file:
                file.remove('.DS_Store')

            if len(file) > 0:
                label_dir = dirname.replace('emotions_dataset', 'emotions_labels')
                if 'emotion.txt' in os.listdir(label_dir):
                    continue

                for label in os.listdir(label_dir):
                    print(label)

                    if label != '.DS_Store':
                        label_f = os.path.join(label_dir, label)
                        class_label = int(float(open(label_f, 'r').readline()[:7].replace(' ', ''))) - 1
                        class_label = map_to[class_map[class_label]]

                for f in file:
                    img_path = os.path.join(dirname, f)
                    img = cv.imread(img_path)
                    img = cv.flip(img, 1)

                    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
                    detector.setInput(blob)
                    detections = detector.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.9:
                            x1 = detections[0, 0, i, 3]
                            y1 = detections[0, 0, i, 4]
                            x2 = detections[0, 0, i, 5]
                            y2 = detections[0, 0, i, 6]

                            # y1 value is increased to look more like microsoft-style prediction
                            coordinates = [x1, y1 * 1.4, x2, y2]

                    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                    hsv[..., 2] = hsv[..., 2] * 0.8
                    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                    pp = os.path.join('/Users/user/Desktop/tmp_ck', f'img{ind}.png')
                    cv.imwrite(pp, img)
                    ind += 1

                    example = get_example(pp, [coordinates], [class_label])
                    writer.write(example.SerializeToString())
    except Exception:
        traceback.print_exc()
    finally:
        writer.close()



if __name__ == '__main__':
    app.run(main)
# try:
#     app.run(main)
# except Exception:
#     print(Exception)