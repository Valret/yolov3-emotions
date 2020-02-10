import sys
sys.path.append('../yolov3-diploma')

import os
import time
import cv2 as cv
import tensorflow as tf
import numpy as np
from core.utils import get_example
from absl import app, flags
from absl.flags import FLAGS
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from tqdm import tqdm

KEY = 'e7d85f97770144068b8cb0135d96f54c'
ENDPOINT = 'https://westcentralus.api.cognitive.microsoft.com'

flags.DEFINE_string('input', './data/FDDB-folds', 'Path to initial dataset')
flags.DEFINE_string('output', './data/fddb_dataset2.tfrecord', 'Path to preprocessed dataset')


def main(args):
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    contents = []
    lengths = []

    for i in range(1, 11):
        with open(os.path.join(FLAGS.input, f'FDDB-fold-{i:02d}-ellipseList.txt')) as file:
            line = file.readline().rstrip()
            while line:
                if not line.endswith('  1'):  # Mode originalPics folder to fddb-folds!
                    contents.append(os.path.join(FLAGS.input, 'originalPics', line) + '.jpg')
                    lengths.append(int(file.readline().rstrip()))

                line = file.readline().rstrip()


    writer = tf.io.TFRecordWriter(FLAGS.output)

    for image_p, length in tqdm(zip(contents, lengths), total=len(contents)):
        try:
            with open(image_p, 'rb') as img:
                detected_faces = face_client.face.detect_with_stream(image=img,
                                                                     return_face_attributes=['emotion'])
        except Exception:
            time.sleep(10)
            detected_faces = None

        if not detected_faces or len(detected_faces) != length:
            continue

        emotion_list, coordinates, class_labels = [], [], []

        image = cv.imread(image_p)
        h, w = image.shape[:2]

        for face in detected_faces:
            face_rect = face.face_rectangle
            x1 = face_rect.left
            y1 = face_rect.top
            x2 = x1 + face_rect.width
            y2 = y1 + face_rect.height

            # Add emotion with highest probability
            emotions = face.face_attributes.emotion.__dict__
            del emotions['additional_properties']
            emotions = np.array([[k, v] for k, v in emotions.items()])

            coordinates.append([x1 / w, y1 / h, x2 / w, y2 / h])
            class_labels.append(np.argmax(emotions[:, 1]))
            emotion_list.append(emotions[np.argmax(emotions[:, 1]), 0])

        example = get_example(image_p, coordinates, class_labels)
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    app.run(main)

#FDDB dataset