import sys
sys.path.append('../yolov3-diploma')

import requests
import os
import numpy as np§
import cv2 as cv
import tensorflow as tf
import json
import time
import traceback
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from absl import app, flags
from absl.flags import FLAGS
from core.utils import get_example
from tqdm import tqdm



KEY = 'e7d85f97770144068b8cb0135d96f54c'
ENDPOINT = 'https://westcentralus.api.cognitive.microsoft.com'

flags.DEFINE_string('input', './data/kaggle.json', 'Path to initial dataset')
flags.DEFINE_string('output', './data/kaggle_dataset2.tfrecord', 'Path to preprocessed dataset')


face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
contents = []


def main(args):
    with open(FLAGS.input, 'r') as file:
        line = file.readline()
        while line:
            data = json.loads(line)
            contents.append((data['content'], len(data['annotation'])))
            line = file.readline()

    writer = tf.io.TFRecordWriter(FLAGS.output)

    for image, length in tqdm(contents):
        try:
            detected_faces = face_client.face.detect_with_url(url=image, return_face_attributes=['emotion'])
        except Exception:
            traceback.print_exc()
            time.sleep(10)
            detected_faces = None

        if not detected_faces or len(detected_faces) != length:
            continue

        coordinates, class_labels = [], []

        response = requests.get(image, stream=True).raw
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)

        # Temporary save current image for get_example method
        tmp_path = os.path.join(FLAGS.input, 'tmp.png')
        cv.imwrite(tmp_path, image)
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

            coordinates.append([x1/w, y1/h, x2/w, y2/h])
            class_labels.append(np.argmax(emotions[:, 1]))

        example = get_example(tmp_path, coordinates, class_labels)
        writer.write(example.SerializeToString())

        # cv.imshow('result', image)

    writer.close()


if __name__ == '__main__':
    app.run(main)