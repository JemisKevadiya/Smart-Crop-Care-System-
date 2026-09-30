"""Validate an uploaded leaf image and turn it into ResNet50 input.

    uploaded file -> validate -> RGB (EXIF-rotated) -> centre square crop
    -> resize 224x224 -> resnet50.preprocess_input -> (1, 224, 224, 3) float32

The resize and preprocess_input steps are the same functions the training
pipeline uses, so a dataset image gives exactly the same tensor here as it did
during validation/testing. Training images are square; non-square uploads are
centre-cropped to a square first so the leaf is not stretched.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.preprocessing.pipeline import resize_image

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_SIDE = 32                      # pixels
MAX_PIXELS = 40_000_000            # guards against decompression bombs
# Largest per-channel pixel standard deviation (on a 64x64 thumbnail) below
# which an image is treated as blank. The least varied of the 52,709 dataset
# images scores 11.8.
MIN_PIXEL_STD = 5.0


class ImageValidationError(ValueError):
    """The input is not a usable image. The message is safe to show to users."""


def _read_bytes(source):
    """Get raw bytes from a path, bytes, or file-like object (e.g. a Streamlit upload)."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ImageValidationError(f"File not found: {path.name}")
        return path.read_bytes()
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        if isinstance(data, str):
            raise ImageValidationError("Expected binary image data, got text.")
        return data
    raise ImageValidationError(f"Unsupported input type: {type(source).__name__}")


def validate_image(source):
    """Return the image as an RGB PIL image, or raise ImageValidationError.

    Accepts a file path, raw bytes, a binary file-like object or a PIL image.
    """
    if isinstance(source, Image.Image):
        image = source
    else:
        data = _read_bytes(source)
        if not data:
            raise ImageValidationError("The file is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise ImageValidationError(
                f"The file is too large ({len(data) / 1e6:.1f} MB); "
                f"the limit is {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()  # structural check without decoding pixels
            image = Image.open(io.BytesIO(data))
            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    f"Unsupported image format '{image.format}'. "
                    f"Use one of: {', '.join(sorted(ALLOWED_FORMATS))}.")
            if image.width * image.height > MAX_PIXELS:
                raise ImageValidationError("The image resolution is too large.")
            image.load()  # full decode: catches truncated/corrupted files
        except ImageValidationError:
            raise
        except (UnidentifiedImageError, Image.DecompressionBombError):
            raise ImageValidationError("The file is not a valid image.") from None
        except (OSError, SyntaxError, ValueError) as exc:
            raise ImageValidationError(f"The image is corrupted or incomplete ({exc}).") from None

    image = ImageOps.exif_transpose(image)  # phone photos store rotation in EXIF
    if min(image.size) < MIN_SIDE:
        raise ImageValidationError(
            f"The image is too small ({image.width}x{image.height}); "
            f"it must be at least {MIN_SIDE}x{MIN_SIDE} pixels.")
    if image.mode != "RGB":
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            # Composite transparency onto white instead of letting it turn black.
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

    # Spatial variation per channel, so a single solid colour counts as blank.
    thumb = np.asarray(image.resize((64, 64)), dtype=np.float32)
    if thumb.std(axis=(0, 1)).max() < MIN_PIXEL_STD:
        raise ImageValidationError(
            "The image looks blank (almost a single colour). Upload a photo of a leaf.")
    return image


def center_square(image):
    """Crop the largest centred square (no-op for square images)."""
    w, h = image.size
    if w == h:
        return image
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess_image(source):
    """Validate and convert an image to a (1, 224, 224, 3) float32 model input."""
    image = center_square(validate_image(source))
    pixels = np.asarray(image, dtype=np.uint8)
    tensor = preprocess_input(resize_image(pixels))
    return tensor.numpy()[np.newaxis, ...]
