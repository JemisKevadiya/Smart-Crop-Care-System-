"""LeafLiteNet architecture, its training-time preprocessing and the cached pipeline."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.model import lite_cnn
from src.preprocessing import config, pipeline, splits

DATA = (config.ROOT / splits.load_split("test")["path"].iloc[0]).exists()


@pytest.fixture(scope="module")
def model():
    return lite_cnn.build_model(38)


def test_shapes_and_size(model):
    assert model.input_shape == (None, 224, 224, 3) and model.output_shape == (None, 38)
    assert model.count_params() < 2_000_000
    out = model(np.zeros((2, 224, 224, 3), "float32"), training=False).numpy()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_ablation_variants_are_smaller():
    full = lite_cnn.build_model(38)
    assert lite_cnn.build_model(38, use_se=False).count_params() < full.count_params()
    assert lite_cnn.build_model(38, width=0.5).count_params() < full.count_params()
    assert lite_cnn.count_macs(lite_cnn.build_model(38, width=0.5)) < lite_cnn.count_macs(full)


def test_training_preprocess_matches_resnet_preprocess_input():
    x = np.random.default_rng(0).uniform(-20, 280, (2, 8, 8, 3)).astype("float32")
    ours = lite_cnn.CaffePreprocess()(x).numpy()
    expected = preprocess_input(np.clip(x, 0, 255).copy())
    np.testing.assert_allclose(ours, expected, atol=1e-4)


def test_training_model_is_identity_at_inference(model):
    """Without augmentation, wrapper(raw) == core(preprocess_input(raw))."""
    wrapper = lite_cnn.build_training_model(model)
    raw = np.random.default_rng(1).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    np.testing.assert_allclose(wrapper(raw, training=False).numpy(),
                               model(preprocess_input(raw.copy()), training=False).numpy(),
                               atol=1e-5)


def test_saved_model_round_trip(model, tmp_path):
    path = tmp_path / "m.keras"
    model.save(path)
    x = np.random.default_rng(2).uniform(-120, 150, (1, 224, 224, 3)).astype("float32")
    np.testing.assert_allclose(keras.models.load_model(path)(x).numpy(), model(x).numpy(),
                               atol=1e-6)


@pytest.mark.skipif(not DATA, reason="dataset not available")
def test_cached_pipeline_gives_identical_images(tmp_path):
    df = splits.load_split("val").head(6)
    plain = pipeline.make_dataset("val", 6, manifest=df)
    cached = pipeline.make_dataset("val", 6, manifest=df, cache=tmp_path / "val.cache")
    raw = pipeline.make_dataset("val", 6, manifest=df, preprocess=False)
    (a, ya), (b, yb), (r, _) = next(iter(plain)), next(iter(cached)), next(iter(raw))
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_array_equal(ya.numpy(), yb.numpy())
    np.testing.assert_allclose(preprocess_input(tf.identity(r)).numpy(), a.numpy(), atol=1e-4)
