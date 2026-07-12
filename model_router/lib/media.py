
"""Image, audio, and video media utilities for model interfaces."""

import base64
import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from PIL import Image


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def image_to_base64_with_mime(image: Image.Image) -> Tuple[str, str]:
    """Convert a PIL Image to (base64_string, mime_type).

    Preserves the original format (WebP, JPEG, PNG) where possible.
    RGBA images are composited onto a white background before JPEG encoding.
    """
    buffer = io.BytesIO()
    format_to_use = "PNG"
    mime_type = "image/png"

    if hasattr(image, "format") and image.format:
        fmt = image.format.upper()
        if fmt == "WEBP":
            format_to_use, mime_type = "WEBP", "image/webp"
        elif fmt in ("JPEG", "JPG"):
            format_to_use, mime_type = "JPEG", "image/jpeg"
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
                image = bg
        elif fmt == "PNG":
            format_to_use, mime_type = "PNG", "image/png"

    if format_to_use == "JPEG":
        image.save(buffer, format=format_to_use, quality=95)
    else:
        image.save(buffer, format=format_to_use)

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime_type


def image_to_base64(image: Image.Image, include_data_uri_prefix: bool = False) -> str:
    """Convert a PIL Image to a base64 string, optionally with a data URI prefix."""
    b64, mime_type = image_to_base64_with_mime(image)
    if include_data_uri_prefix:
        return f"data:{mime_type};base64,{b64}"
    return b64


def base64_to_image(base64_string: str) -> Image.Image:
    """Decode a base64 string (with or without data URI prefix) to a PIL Image."""
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(base64_string)))


def detect_image_mime_type(base64_string: str) -> str:
    """Detect image MIME type from base64 data using magic bytes.

    Falls back to image/png for unknown formats.
    """
    if base64_string.startswith("data:"):
        mime_part = base64_string.split(";")[0]
        if mime_part.startswith("data:"):
            return mime_part[5:]

    try:
        partial = base64.b64decode(base64_string[:16])
    except Exception:
        return "image/png"

    if partial[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if partial[:2] == b"\xff\xd8":
        return "image/jpeg"
    if partial[:4] == b"RIFF" and partial[8:12] == b"WEBP":
        return "image/webp"
    if partial[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


def convert_query_images_to_base64_list(
    images: Union[str, Image.Image, List, dict],
    include_data_uri_prefix: bool = False,
) -> List[str]:
    """Normalise a Query.images value to a flat list of base64 strings."""
    if isinstance(images, str):
        return [images]
    if isinstance(images, Image.Image):
        return [image_to_base64(images, include_data_uri_prefix)]
    if isinstance(images, list):
        return [
            image_to_base64(img, include_data_uri_prefix) if isinstance(img, Image.Image) else img
            for img in images
        ]
    if isinstance(images, dict):
        result = []
        for img in images.values():
            if isinstance(img, Image.Image):
                result.append(image_to_base64(img, include_data_uri_prefix))
            elif isinstance(img, list):
                result.extend(convert_query_images_to_base64_list(img))
            elif isinstance(img, str):
                result.append(img)
            else:
                raise ValueError(f"Unsupported image type {type(img)!r}")
        return result
    raise ValueError(f"Unsupported images format: {type(images)!r}")


def compress_for_reference(
    images: list,
    max_px: int = 1024,
    jpeg_quality: int = 85,
) -> list[str]:
    """Downscale and JPEG-compress PIL images for WebSocket-friendly payloads.

    Returns data-URI base64 strings. Plain strings are passed through unchanged.
    """
    result = []
    for img in images:
        if isinstance(img, Image.Image):
            w, h = img.size
            if max(w, h) > max_px:
                scale = max_px / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            result.append(f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}")
        elif isinstance(img, str):
            result.append(img)
    return result


def save_image_to_temp(image: Image.Image, prefix: str = "demo_image") -> str:
    """Save a PIL Image to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix=f"{prefix}_")
    os.close(fd)
    image.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def audio_to_base64(audio_data: bytes) -> str:
    """Encode raw audio bytes to a base64 string."""
    return base64.b64encode(audio_data).decode("utf-8")


def base64_to_audio(base64_string: str) -> bytes:
    """Decode a base64 string (with or without data URI prefix) to audio bytes."""
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    return base64.b64decode(base64_string)


def save_audio_to_temp(
    audio_data: bytes, format: str = "mp3", prefix: str = "demo_audio"
) -> str:
    """Save audio bytes to a temp file and return the path.

    The format string may be compound (e.g. "mp3_44100_128") — only the
    first segment is used as the file extension.
    """
    extension = format.split("_")[0] if "_" in format else format
    fd, path = tempfile.mkstemp(suffix=f".{extension}", prefix=f"{prefix}_")
    try:
        os.write(fd, audio_data)
    finally:
        os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def save_video_to_temp(
    video_data: bytes, format: str = "mp4", prefix: str = "demo_video"
) -> str:
    """Save video bytes to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=f".{format}", prefix=f"{prefix}_")
    try:
        os.write(fd, video_data)
    finally:
        os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Outpainting geometry
# ---------------------------------------------------------------------------

@dataclass
class OutpaintingExtent:
    """Represents the pixel extents of an outpainting operation.

    Handles images larger than the model's maximum dimension by scaling
    the working space down before generation and back up afterwards.
    """

    top: int
    bottom: int
    left: int
    right: int

    original_size: Tuple[int, int]
    target_size: Tuple[int, int]

    working_input_size: Tuple[int, int]
    """Input image size in the scaled working space."""

    scaled_output_size: Tuple[int, int]
    """Output canvas size in the scaled working space."""

    pre_scale: float
    """Scale applied to the input before generation (< 1.0 means downscale)."""

    post_scale: float
    """Scale applied to the output after generation (> 1.0 means upscale)."""

    @classmethod
    def from_image_sizes(
        cls,
        image_size: Tuple[int, int],
        target_image_size: Tuple[int, int],
        image_position: Optional[Tuple[int, int]] = None,
        max_dimension: int = 2048,
        divisibility: int = 64,
    ) -> "OutpaintingExtent":
        """Calculate outpainting extents with automatic scaling for large images.

        Args:
            image_size: Input image dimensions (width, height).
            target_image_size: Desired output canvas dimensions (can exceed max_dimension).
            image_position: Where the original image sits on the canvas (x, y).
                            Defaults to centred.
            max_dimension: Model's maximum supported dimension.
            divisibility: Required alignment for all dimensions.
        """
        orig_w, orig_h = image_size
        tgt_w, tgt_h = target_image_size

        scale = 1.0
        if tgt_w > max_dimension or tgt_h > max_dimension:
            scale = min(max_dimension / tgt_w, max_dimension / tgt_h)

        wk_tgt_w = cls._round_to_divisible(int(tgt_w * scale), divisibility)
        wk_tgt_h = cls._round_to_divisible(int(tgt_h * scale), divisibility)
        wk_in_w = int(orig_w * scale)
        wk_in_h = int(orig_h * scale)

        wk_pos = None if image_position is None else (
            int(image_position[0] * scale), int(image_position[1] * scale)
        )
        extents = cls._calculate_extents(
            (wk_in_w, wk_in_h), (wk_tgt_w, wk_tgt_h), wk_pos, divisibility
        )

        return cls(
            top=extents["top"], bottom=extents["bottom"],
            left=extents["left"], right=extents["right"],
            original_size=image_size, target_size=target_image_size,
            working_input_size=(wk_in_w, wk_in_h),
            scaled_output_size=(wk_tgt_w, wk_tgt_h),
            pre_scale=scale,
            post_scale=1.0 / scale if scale > 0 else 1.0,
        )

    @staticmethod
    def _round_to_divisible(value: int, divisibility: int) -> int:
        return round(value / divisibility) * divisibility

    @staticmethod
    def _calculate_extents(
        image_size: Tuple[int, int],
        target_size: Tuple[int, int],
        image_position: Optional[Tuple[int, int]],
        divisibility: int,
    ) -> dict:
        w, h = image_size
        tw, th = target_size
        pad_x = max(0, tw - w)
        pad_y = max(0, th - h)

        if image_position is None:
            left, right = pad_x // 2, pad_x - pad_x // 2
            top, bottom = pad_y // 2, pad_y - pad_y // 2
        else:
            x, y = max(0, image_position[0]), max(0, image_position[1])
            left, top = x, y
            right = max(0, tw - w - x)
            bottom = max(0, th - h - y)

        # Ceil-round each extent to the divisibility boundary
        left   = math.ceil(left   / divisibility) * divisibility
        right  = math.ceil(right  / divisibility) * divisibility
        top    = math.ceil(top    / divisibility) * divisibility
        bottom = math.ceil(bottom / divisibility) * divisibility

        return {"top": top, "bottom": bottom, "left": left, "right": right}

    @property
    def needs_preprocessing(self) -> bool:
        return self.pre_scale < 1.0

    @property
    def needs_postprocessing(self) -> bool:
        return self.post_scale > 1.0
