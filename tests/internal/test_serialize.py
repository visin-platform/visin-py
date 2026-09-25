import enum
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from visin._internal.serialize import to_jsonable


class NumpyScalar:
    """Behaves like np.float32: tolist() gives a Python float."""

    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class Tensor:
    """Behaves like a 0-d torch tensor without tolist."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def test_plain_json_is_unchanged():
    value = {"a": [1, 2.5, "x", None, True], "b": {"c": False}}
    assert to_jsonable(value) == value


def test_array_likes_become_numbers():
    assert to_jsonable({"loss": NumpyScalar(0.25), "iou": Tensor(0.5)}) == {"loss": 0.25, "iou": 0.5}


def test_arrays_become_lists():
    assert to_jsonable(NumpyScalar([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_nan_and_infinity_become_null_and_are_reported():
    found = []
    assert to_jsonable({"val": {"loss": math.nan, "iou": [math.inf]}}, found) == {
        "val": {"loss": None, "iou": [None]}
    }
    assert found == ["val.loss", "val.iou[0]"]


def test_a_nan_inside_an_array_like_is_also_caught():
    found = []
    assert to_jsonable({"x": NumpyScalar(math.nan)}, found) == {"x": None}
    assert found == ["x"]


def test_common_standard_types():
    class Colour(enum.Enum):
        RED = "red"

    @dataclass
    class Point:
        x: int

    stamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
    ident = uuid.uuid4()
    assert to_jsonable(
        {
            "d": Decimal("0.5"),
            "t": stamp,
            "u": ident,
            "p": Path("a/b"),
            "e": Colour.RED,
            "dc": Point(1),
            "s": (1, 2),
        }
    ) == {
        "d": 0.5,
        "t": stamp.isoformat(),
        "u": str(ident),
        "p": str(Path("a/b")),
        "e": "red",
        "dc": {"x": 1},
        "s": [1, 2],
    }


def test_keys_become_strings():
    assert to_jsonable({1: "a"}) == {"1": "a"}


def test_an_unknown_object_names_where_it_was():
    with pytest.raises(TypeError, match=r"at model\.loss"):
        to_jsonable({"model": {"loss": object()}})


def test_a_fallback_converts_what_nothing_else_can():
    class Loss:
        def __repr__(self):
            return "DiceLoss()"

    assert to_jsonable({"loss": Loss()}, fallback=repr) == {"loss": "DiceLoss()"}
