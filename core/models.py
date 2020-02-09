import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers, losses
from .utils import get_anchors


class DarknetBlock(Model):
    def __init__(self, filters, size, strides=1, batch_norm=True):  # Order of layers definition is crucial!
        super(DarknetBlock, self).__init__()
        self.conv2d_0 = layers.Conv2D(filters=filters, kernel_size=size, strides=strides, padding='same',
                                      use_bias=(not batch_norm), kernel_regularizer=regularizers.l2(0.0005))
        self.bn_0 = BatchNormalization() if batch_norm else None
        self.lrelu_0 = layers.LeakyReLU(alpha=0.1) if batch_norm else None

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class Darknet19(Model):
    def __init__(self, name=None):
        super(Darknet19, self).__init__(name=name)  # Order of layers definition is crucial!
        for f in [16, 32, 64, 128, 256, 512]:
            setattr(self, f'darkconv_{f}', DarknetBlock(f, 3))
            setattr(self, f'maxpool_{f}', layers.MaxPool2D(2, (2 if f != 512 else 1), 'same'))
        self.darkconv_1024 = DarknetBlock(1024, 3)

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            if i != 8:
                x = layer(x)
            else:
                x = x_skip = layer(x)
        return x, x_skip


class YoloBlock(Model):
    def __init__(self, filters, name=None):
        super(YoloBlock, self).__init__(name=name)
        self.darkconv_0 = DarknetBlock(filters, 1)
        self.upsample_0 = layers.UpSampling2D(2)
        self.concat_0 = layers.Concatenate()

    def call(self, inputs):
        x, x_conc = inputs
        x = self.darkconv_0(x)
        x = self.upsample_0(x)
        x = self.concat_0([x, x_conc])
        return x


class YoloOutput(Model):
    def __init__(self, filters, anchors, classes, name=None):
        super(YoloOutput, self).__init__(name=name)
        self.darknet_0 = DarknetBlock(filters * 2, 3)
        self.darknet_1 = DarknetBlock(anchors * (classes + 5), 1, batch_norm=False)
        self.lambda_0 = layers.Lambda(lambda x:
                                      tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))

    def call(self, inputs):
        x = self.darknet_0(inputs)
        x = self.darknet_1(x)
        x = self.lambda_0(x)
        return x


class YoloAux(Model):
    def __init__(self, filters, anchors, classes, name=None):
        super(YoloAux, self).__init__(name=name)
        self.deepthwise_0 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=4)
        self.bn_0 = BatchNormalization()
        self.lrelu_0 = layers.LeakyReLU(alpha=0.1)
        self.deepthwise_1 = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=4)
        self.bn_1 = BatchNormalization()
        self.lrelu_1  = layers.LeakyReLU(alpha=0.1)
        self.deepthwise_2 = layers.DepthwiseConv2D(kernel_size=8, strides=8, padding='valid', depth_multiplier=4)
        self.bn_2 = BatchNormalization()
        self.lrelu_2  = layers.LeakyReLU(alpha=0.1)
        self.concat_0 = layers.Concatenate()

    def call(self, inputs):
        x = self.darknet_0(inputs)
        x = self.darknet_1(x)
        x = self.lambda_0(x)
        return x

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if training is None:
            training = False
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def Yolo(size=448, channels=3, classes=8, iou_tres=0.5, score_tres=0.05, max_total=20, training=False, aux=False):
    anchors, mask_len = get_anchors(size)
    x = inputs = layers.Input([size, size, channels])

    x, x_skip = Darknet19(name='darknet_recursive')(x)

    if aux:
        x_residual = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=4)(inputs)
        x_residual = BatchNormalization()(x_residual)
        x_residual = layers.LeakyReLU(alpha=0.1)(x_residual)

        x_residual = layers.DepthwiseConv2D(kernel_size=2, strides=2, padding='valid', depth_multiplier=4)(x_residual)
        x_residual = BatchNormalization()(x_residual)
        x_residual = layers.LeakyReLU(alpha=0.1)(x_residual)

        x_residual = layers.DepthwiseConv2D(kernel_size=8, strides=8, padding='valid', depth_multiplier=4)(x_residual)
        x_residual = BatchNormalization()(x_residual)
        x_residual = layers.LeakyReLU(alpha=0.1)(x_residual)
        x = layers.Concatenate()([x, x_residual])

    x = DarknetBlock(256, 1)(x)
    output_0 = YoloOutput(256, mask_len, classes, name='output_recursive_0')(x)

    x = YoloBlock(128, name='block_recursive_0')((x, x_skip))
    output_1 = YoloOutput(128, mask_len, classes, name='output_recursive_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3_train')

    boxes_0 = make_bboxs(output_0, anchors[:mask_len], classes)
    boxes_1 = make_bboxs(output_1, anchors[mask_len:], classes)
    outputs = nms(boxes_0, boxes_1, max_total, iou_tres, score_tres, classes)
    return Model(inputs, outputs, name='yolov3_inference')


def make_bboxs(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    xy, wh, probs, type_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    xy = tf.sigmoid(xy)
    probs = tf.sigmoid(probs)
    type_probs = tf.sigmoid(type_probs)
    orig_pred = tf.concat((xy, wh), axis=-1)  # original xywh for loss

    # Cartesian distribution over the grid, used Lambda wrapper because of tf 2.0 bug with meshgrid
    grid = layers.Lambda(lambda x: tf.meshgrid(tf.range(x), tf.range(x)))(grid_size)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=-2)  # Push empty -2 dimension to match xy
    # grid = tf.tile(grid, [1, 1, 3, 1]) Tensorflow automatically do this in the next step

    # grid_xy represent shifts for each grid cell coordinate:
    grid_xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    wh = tf.exp(wh) * anchors  # Tune anchor sizes

    top_l = grid_xy - wh / 2
    bottom_r = grid_xy + wh / 2
    bbox = tf.concat([top_l, bottom_r], axis=-1)

    return bbox, probs, type_probs, orig_pred


def nms(output_1, output_2, max_total, iou_tres, score_tres, classes):
    # 0 - bboxs, 1 - probs, 2 - type_probs
    tmp = []

    for o_1, o_2 in zip(output_1, output_2):
        tmp.append(tf.concat([layers.Flatten()(o_1), layers.Flatten()(o_2)], axis=-1))

    probs = tf.reshape(tmp[1], (tf.shape(tmp[1])[0], -1, 1))
    type_probs = tf.reshape(tmp[2], (tf.shape(tmp[2])[0], -1, classes))
    scores = type_probs * probs

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(tmp[0], (tf.shape(tmp[0])[0], -1, 1, 4)),
        scores=scores,
        max_output_size_per_class=max_total,
        max_total_size=max_total,
        iou_threshold=iou_tres,
        score_threshold=score_tres
    )

    return boxes, scores, classes, valid_detections


def bbox_iou(pred, true):
    # Compose new shape from biggest dimensions of both
    pred = tf.expand_dims(pred, -2)  # -2 dimension will be same as 0 dimension of true
    new_shape = tf.broadcast_dynamic_shape(tf.shape(pred), tf.shape(true))

    # Give inputs that shape, copy values across empty dimensions
    pred = tf.broadcast_to(pred, new_shape)
    true = tf.broadcast_to(true, new_shape)

    # Calculate longest width and height for intersection >= 0
    longest_w = tf.maximum(0, tf.minimum(pred[..., 2], true[..., 2]) - tf.maximum(pred[..., 0], true[..., 0]))
    longest_h = tf.maximum(0, tf.minimum(pred[..., 3], true[..., 3]) - tf.maximum(pred[..., 1], true[..., 1]))
    biggest_area = longest_w * longest_h

    # Calculate areas of true and pred bboxes to get their union
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    true_area = (true[..., 2] - true[..., 0]) * (true[..., 3] - true[..., 1])
    best_ious = tf.reduce_max(biggest_area / (pred_area + true_area - biggest_area), axis=-1)

    return best_ious


def Loss(anchors, classes=8, iou_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # y_pred and y_true have shapes (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        p_bboxes, p_objectness, p_class, p_original = make_bboxs(y_pred, anchors, classes)
        p_xy, p_wh = tf.split(p_original, (2, 2), axis=-1)

        # t_bboxes, t_objectness, t_class_ids = tf.split(y_true, (4, 1, 1), axis=-1)
        t_xy = (y_true[..., 0:2] + y_true[..., 2:4]) / 2  # Center of each coord
        t_wh = y_true[..., 2:4] - y_true[..., 0:2]  # Width and height for each

        # Give higher weights to small boxes
        l_tune = 2 - t_wh[..., 0] * t_wh[..., 1]

        # Inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        t_xy = t_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        t_wh = tf.math.log(t_wh / anchors)
        t_wh = tf.where(tf.math.is_inf(t_wh), tf.zeros_like(t_wh), t_wh)

        # Calculate masks
        obj_mask = tf.squeeze(y_true[..., 4:5], -1)
        gather_mask = tf.where(tf.not_equal(obj_mask, 0.))

        buffer = [0] * 7
        for i, el in enumerate([t_wh, p_wh, t_xy, p_xy, y_true[..., 5:6], p_class, l_tune]):
            buffer[i] = tf.gather_nd(el, gather_mask)

        t_wh, p_wh, t_xy, p_xy, t_class, p_class, l_tune = buffer

        # Calculate best IOU between predictions and true objects
        best_iou = bbox_iou(p_bboxes, tf.boolean_mask(y_true[..., :4], tf.cast(obj_mask, tf.bool)))
        obj_loss = losses.binary_crossentropy(y_true[..., 4:5], p_objectness)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * obj_loss * tf.cast(best_iou < iou_thresh, tf.float32)

        # Losses
        wh_loss = l_tune * tf.reduce_sum(tf.square(t_wh - p_wh), axis=-1)
        xy_loss = l_tune * tf.reduce_sum(tf.square(t_xy - p_xy), axis=-1)

        class_loss = losses.sparse_categorical_crossentropy(t_class, p_class) * 20

        total_loss = 0
        for loss in [xy_loss, wh_loss, obj_loss, class_loss]:
            total_loss += tf.reduce_sum(loss)
        return total_loss
    return yolo_loss
