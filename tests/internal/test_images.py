import struct
import zlib

import numpy as np
import pytest

from visin._internal.images import encode_png, is_image


def decode(png):
    """Width, height, colour type and the unfiltered pixel bytes of a PNG this module wrote."""
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    position, idat, header = 8, b"", None
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        kind = png[position + 4 : position + 8]
        data = png[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", png[position + 8 + length : position + 12 + length])
        assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            idat += data
        position += 12 + length
    width, height, _, color, *_ = header
    channels = {0: 1, 4: 2, 2: 3, 6: 4}[color]
    raw = zlib.decompress(idat)
    stride = width * channels + 1
    pixels = b"".join(raw[row * stride + 1 : (row + 1) * stride] for row in range(height))
    return width, height, color, np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, channels)


def test_a_uint8_rgb_array_round_trips():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    width, height, color, pixels = decode(encode_png(image))
    assert (width, height, color) == (3, 2, 2)
    assert (pixels == image).all()


def test_a_grayscale_array_is_one_channel():
    width, height, color, _ = decode(encode_png(np.zeros((4, 5), dtype=np.uint8)))
    assert (width, height, color) == (5, 4, 0)


def test_floats_up_to_one_are_scaled_and_larger_ones_are_taken_as_they_are():
    assert decode(encode_png(np.array([[0.0, 0.5, 1.0]])))[3].ravel().tolist() == [0, 128, 255]
    assert decode(encode_png(np.array([[0.0, 100.0, 300.0]])))[3].ravel().tolist() == [0, 100, 255]


def test_a_boolean_mask_is_black_and_white():
    assert decode(encode_png(np.array([[True, False]])))[3].ravel().tolist() == [255, 0]


def test_channels_first_is_turned_to_channels_last():
    chw = np.zeros((3, 8, 10), dtype=np.uint8)
    width, height, color, _ = decode(encode_png(chw))
    assert (width, height, color) == (10, 8, 2)


def test_a_tensor_like_object_is_read_through_detach_cpu_numpy():
    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.full((2, 2), 7, dtype=np.uint8)

    assert decode(encode_png(Tensor()))[3].ravel().tolist() == [7, 7, 7, 7]


def test_a_matplotlib_figure_and_a_pil_image_are_saved_by_their_own_methods():
    class Figure:
        def savefig(self, buffer, format):  # noqa: A002
            buffer.write(b"fig:" + format.encode())

    class Picture:
        mode = "RGB"

        def save(self, buffer, format):  # noqa: A002
            buffer.write(b"pil:" + format.encode())

    assert encode_png(Figure()) == b"fig:png"
    assert encode_png(Picture()) == b"pil:PNG"


def test_an_image_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="cannot make an image"):
        encode_png(np.zeros((2, 3, 5, 5)))


def test_paths_are_not_images():
    assert not is_image("a.png")
    assert not is_image(__import__("pathlib").Path("a.png"))
    assert is_image(np.zeros((1, 1)))
