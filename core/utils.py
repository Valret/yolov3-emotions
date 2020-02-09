from absl import logging
import numpy as np
import tensorflow as tf
import cv2 as cv
import colorsys

anchors = np.array([(81, 82), (135, 169), (220, 300),
                         (10, 14), (23, 27), (37, 58)], np.float32)
mask_len = 3

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def define_basics(flags):
    flags.DEFINE_string('classes', './data/faces.names', 'Path to classes names file')
    flags.DEFINE_string('weights', './checkpoints/yolov3_pc.tf', 'Path to weights file')
    flags.DEFINE_integer('size', 448, 'Image size')


def get_example(img_path, coordinates, class_labels):
    img_raw = open(img_path, 'rb').read()
    xmin, ymin, xmax, ymax, classes = [], [], [], [], []

    for c, l in zip(coordinates, class_labels):
        xmin.append(float(c[0]))
        ymin.append(float(c[1]))
        xmax.append(float(c[2]))
        ymax.append(float(c[3]))
        classes.append(l)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

    return example


def load_darknet_weights(model, weights_file, tiny=True):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    ind = 0

    final_val = 0

    for sub_model in model.layers:
        iterator = enumerate(sub_model.layers) if '_recursive' in sub_model.name \
                                               else enumerate([sub_model])
        for i, layer_out in iterator:
            if not 'conv_block' in layer_out.name:
                print(layer_out.name)
            else:
                for i, layer in enumerate(layer_out.layers):
                    ind += 1
                    print('     ' + layer.name)
                    if not layer.name.startswith('conv2d'):
                        continue
                    batch_norm = None
                    if i + 1 < len(layer_out.layers) and \
                            layer_out.layers[i + 1].name.startswith('batch_norm'):
                        batch_norm = layer_out.layers[i + 1]

                    # logging.info("{}/{} {}".format(
                    #     layer_out.name, layer.name, 'bn' if batch_norm else 'bias'))

                    filters = layer.filters
                    size = layer.kernel_size[0]
                    in_dim = layer.get_input_shape_at(0)[-1]

                    if batch_norm is None:
                        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                        final_val += filters
                        print(filters)
                    else:
                        # darknet [beta, gamma, mean, variance]
                        bn_weights = np.fromfile(
                            wf, dtype=np.float32, count=4 * filters)
                        final_val += 4 * filters
                        print(4*filters)
                        # tf [gamma, beta, mean, variance]
                        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                    # darknet shape (out_dim, in_dim, height, width)
                    conv_shape = (filters, in_dim, size, size)
                    conv_weights = np.fromfile(
                        wf, dtype=np.float32, count=np.product(conv_shape))
                    # tf shape (height, width, in_dim, out_dim)
                    conv_weights = conv_weights.reshape(
                        conv_shape).transpose([2, 3, 1, 0])

                    if batch_norm is None:
                        layer.set_weights([conv_weights, conv_bias])
                    else:
                        layer.set_weights([conv_weights])
                        batch_norm.set_weights(bn_weights)

    print(final_val)
    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def get_anchors(size=416):
    print(size)
    return anchors / size, mask_len


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    bbox_thick = (wh[0] + wh[1]) // 1000

    if 'numpy.int32' not in str(type(nums)):
        nums = nums.numpy()

    colors = [colorsys.hsv_to_rgb(1.0 * x / nums, 0.8, 1.0) for x in range(nums)]
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i in range(nums):
        bbox_color = colors[i]
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        img = cv.rectangle(img, x1y1, x2y2, bbox_color, bbox_thick)
        class_text = '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i])
        text_size = cv.getTextSize(class_text, 0, 0.8, thickness=bbox_thick // 2)[0]
        cv.rectangle(img, x1y1, (x1y1[0] + text_size[0], x1y1[1] - int(text_size[1] * 1.3)), bbox_color, -1)  # filled

        img = cv.putText(img, class_text, (x1y1[0], abs(x1y1[1] - 3)),
                         cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), bbox_thick // 2, lineType=cv.LINE_AA)
    return img


def check_dataset(datatest, class_names, n=20):
    for d in datatest.take(n):
        img = d[0].numpy().astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = draw_outputs(img, d[1], class_names)

        cv.imshow('Check', img)
        key = cv.waitKey(2000)
        if key == ord('q'):
            break


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
