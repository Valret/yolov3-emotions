import tensorflow as tf
from absl.flags import FLAGS


@tf.function
def to_scale(y_true, grid_size, mask_len, second):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, mask_len, 6))

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue

            anchor_ind = tf.cast(y_true[i][j][5], tf.int32)  # Matching anchor index
            is_match = tf.less(anchor_ind, mask_len)
            is_match = not is_match if second else is_match
            anchor_ind %= mask_len

            if is_match:
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2  # Center of the box

                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)  # Calculate grid cell coords for this bbox

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_ind])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, mask_len, size):
    grid_size = size // 32

    # Calculate anchor ind
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]

    # Repeat box_wh elements for each different anchor
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))

    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    # Index of the anchor with the biggest IOU for each object
    anchor_ind = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_ind = tf.expand_dims(anchor_ind, axis=-1)

    y_train = tf.concat([y_train, anchor_ind], axis=-1)

    y_train = (to_scale(y_train, grid_size, mask_len, second=False),
               to_scale(y_train, grid_size * 2, mask_len, second=True))

    return y_train


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


IMAGE_FEATURE_MAP = {
    'image/data': tf.io.FixedLenFeature([], tf.string),
    'image/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/label': tf.io.VarLenFeature(tf.int64),
}


def parse_tfrecord(tfrecord, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/data'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_label = tf.cast(tf.sparse.to_dense(
        x['image/label']), tf.float32)

    y_train = tf.stack([tf.sparse.to_dense(x['image/bbox/xmin']),
                        tf.sparse.to_dense(x['image/bbox/ymin']),
                        tf.sparse.to_dense(x['image/bbox/xmax']),
                        tf.sparse.to_dense(x['image/bbox/ymax']),
                        class_label], axis=1)

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, size):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, size))

def load_dummy_validation():
    x_train = tf.zeros([1, 448, 448, 3])
    y_train = tf.zeros([1, 8, 5])

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


def load_one_validation():
    x_train = tf.image.decode_image(
        open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 1],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 2]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
