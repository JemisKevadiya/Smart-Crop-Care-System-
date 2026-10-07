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


def load_image(path):
    """Read one image file into a float32 [224, 224, 3] tensor in [0, 255]."""
    data = tf.io.read_file(path)
    # INTEGER_ACCURATE matches libjpeg's default IDCT (as used by Pillow); TF's
    # default fast IDCT differs by several intensity levels.
    image = tf.io.decode_jpeg(data, channels=config.CHANNELS, dct_method="INTEGER_ACCURATE")
    return resize_image(image)


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
                 manifest=None):
    """Build a batched, prefetched dataset of (preprocessed image, label index).

    `training` defaults to True only for the "train" split: shuffling and
    augmentation are never applied to val/test.
    """
    if training is None:
        training = split == "train"
    df = load_split(split) if manifest is None else manifest
    paths = [str(config.ROOT / p) for p in df["path"]]
    labels = df["label_idx"].to_numpy("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    if training:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.clip_by_value(x, 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def preprocess_image_file(path):
    """Preprocess a single image for inference, identical to val/test."""
    return preprocess_input(load_image(tf.constant(str(path))))[tf.newaxis, ...]


def to_display(batch):
    """Undo preprocess_input for plotting: BGR mean-subtracted -> RGB uint8."""
    rgb = (batch + IMAGENET_BGR_MEAN)[..., ::-1]
    return tf.cast(tf.clip_by_value(rgb, 0, 255), tf.uint8)
