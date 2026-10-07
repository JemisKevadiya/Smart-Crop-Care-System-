"""tf.data input pipeline for ResNet50.

Every split goes through the same deterministic steps:

    read JPEG -> decode RGB -> resize 256x256 -> 224x224 (bilinear, antialiased)
    -> float32 in [0, 255] -> resnet50.preprocess_input

Training additionally shuffles and applies random augmentation *before*
preprocess_input (the augmentation layers work in the [0, 255] pixel range).

preprocess_input uses fixed ImageNet channel means ("caffe" mode: RGB->BGR,
subtract per-channel mean, no scaling), so no statistic is fitted on any split
and validation/test data cannot influence the transform.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

from . import config
from .splits import load_split

AUTOTUNE = tf.data.AUTOTUNE

# ImageNet BGR means subtracted by resnet50.preprocess_input.
IMAGENET_BGR_MEAN = tf.constant([103.939, 116.779, 123.68])


def decode_image(path):
    """Read one image file into a uint8 RGB tensor at its stored size."""
    data = tf.io.read_file(path)
    # INTEGER_ACCURATE matches libjpeg's default IDCT (as used by Pillow); TF's
    # default fast IDCT differs by several intensity levels.
    return tf.io.decode_jpeg(data, channels=config.CHANNELS, dct_method="INTEGER_ACCURATE")


def load_image(path):
    """Read one image file into a float32 [224, 224, 3] tensor in [0, 255]."""
    return resize_image(decode_image(path))


def resize_image(image):
    """Resize a decoded uint8 RGB image to float32 [224, 224, 3] in [0, 255].

    Shared by training and inference so both resize identically.
    """
    image = tf.image.resize(image, config.IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image.set_shape((*config.IMAGE_SIZE, config.CHANNELS))
    return image


def build_augmenter(seed=config.SEED):
    """Random geometric and photometric transforms, active only in training."""
    a = config.AUGMENT
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(a["rotation"], fill_mode="reflect", seed=seed),
            keras.layers.RandomZoom(a["zoom"], fill_mode="reflect", seed=seed),
            keras.layers.RandomTranslation(a["translation"], a["translation"],
                                           fill_mode="reflect", seed=seed),
            keras.layers.RandomBrightness(a["brightness"], value_range=(0, 255), seed=seed),
            keras.layers.RandomContrast(a["contrast"], seed=seed),
        ],
        name="augmentation",
    )


def make_dataset(split, batch_size=config.BATCH_SIZE, training=None, seed=config.SEED,
                 manifest=None, augment=None, preprocess=True, cache=None,
                 shuffle_buffer=4096):
    """Build a batched, prefetched dataset of (preprocessed image, label index).

    `training` defaults to True only for the "train" split: shuffling and
    augmentation are never applied to val/test.

    Options for training elsewhere (e.g. scripts/train_lite_cnn.py on a GPU):
    `augment=False` leaves augmentation to the model, `preprocess=False` returns
    images in [0, 255], and `cache` (a file path) stores the decoded images on
    disk after the first epoch. With a cache, the file order is shuffled once and
    each epoch is shuffled through a `shuffle_buffer`-sized buffer.
    """
    if training is None:
        training = split == "train"
    if augment is None:
        augment = training
    df = load_split(split) if manifest is None else manifest
    paths = [str(config.ROOT / p) for p in df["path"]]
    labels = df["label_idx"].to_numpy("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=cache is None)
    if cache is None:
        ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda p, y: (decode_image(p), y), num_parallel_calls=AUTOTUNE)
        ds = ds.cache(str(cache))
        if training:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (resize_image(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    if augment:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.clip_by_value(x, 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)

    if preprocess:
        ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def preprocess_image_file(path):
    """Preprocess a single image for inference, identical to val/test."""
    return preprocess_input(load_image(tf.constant(str(path))))[tf.newaxis, ...]


def to_display(batch):
    """Undo preprocess_input for plotting: BGR mean-subtracted -> RGB uint8."""
    rgb = (batch + IMAGENET_BGR_MEAN)[..., ::-1]
    return tf.cast(tf.clip_by_value(rgb, 0, 255), tf.uint8)
