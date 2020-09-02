from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

from core.models import Yolo, Loss
from core.utils import freeze_all, check_dataset, get_anchors, define_basics
import core.dataset as dataset

define_basics(flags)
flags.DEFINE_boolean('freeze', False, 'Whether to freeze base part of the model')
flags.DEFINE_boolean('check', False, 'Check current dataset')
flags.DEFINE_boolean('pretrained', False, 'Use pre-trained model for transfer learning')
flags.DEFINE_integer('yolo_max_boxes', 20, 'Maximum number of predictions per image')
flags.DEFINE_string('dataset', './datasets/*.tfrecord', 'Path to dataset')
flags.DEFINE_string('validation_dataset', '', 'Path to validation dataset, use dummy dataset if empty')
flags.DEFINE_integer('epochs', 8, 'Number of epochs')
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('num_classes', 8, 'Number of classes in the model')
flags.DEFINE_integer('weights_num_classes', 80, 'Number of classes in the pre-trained model')



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = Yolo(FLAGS.size, training=True, classes=FLAGS.num_classes, aux=True)
    anchors, mask_len = get_anchors(FLAGS.size)

    train_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.size)

    if FLAGS.check:
        logging.info('Current dataset check')
        with open(FLAGS.classes) as classes_f:
            class_names = [c.strip() for c in classes_f.readlines()]
        logging.info('Classes loaded')
        check_dataset(train_dataset, class_names)

    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.shuffle(buffer_size=200)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, mask_len, FLAGS.size)))

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.validation_dataset:
        validation_dataset = dataset.load_tfrecord_dataset(FLAGS.validation_dataset, FLAGS.classes, FLAGS.size)
    else:
        validation_dataset = dataset.load_dummy_validation()

    validation_dataset = validation_dataset.batch(FLAGS.batch_size)
    validation_dataset = validation_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, mask_len, FLAGS.size)))

    if FLAGS.weights:
        if FLAGS.pretrained:
            model_pretrained = Yolo(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes)
            model_pretrained.load_weights(FLAGS.weights)

            model.get_layer('darknet_recursive').set_weights(
                model_pretrained.get_layer('darknet_recursive').get_weights())
        else:
            model.load_weights(FLAGS.weights)

    if FLAGS.freeze:
        freeze_all(model.get_layer('darknet_recursive'))

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [Loss(anchors[:mask_len], classes=FLAGS.num_classes),
            Loss(anchors[mask_len:], classes=FLAGS.num_classes)]

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True, metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(verbose=1, patience=10),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_face.tf',
                        verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(train_dataset,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks,
                        validation_data=validation_dataset,
                        steps_per_epoch=10)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
