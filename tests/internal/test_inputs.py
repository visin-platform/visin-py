import argparse
import dataclasses

import pytest

from visin._internal.inputs import as_mapping, default_name, epoch_number, merge_results


def test_one_epoch_has_one_number_whatever_type_names_it():
    assert epoch_number(3) == epoch_number(3.0) == 3
    assert isinstance(epoch_number(3.0), int)
    assert epoch_number(2.5) == 2.5


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
def test_an_epoch_must_be_a_finite_number(bad):
    with pytest.raises((TypeError, ValueError)):
        epoch_number(bad)


def test_train_and_val_fold_into_results():
    merged = merge_results({"train": {"loss": 1}, "lr": 0.1}, {"acc": 0.5}, {"loss": 2})
    assert merged == {"train": {"loss": 1, "acc": 0.5}, "lr": 0.1, "val": {"loss": 2}}


def test_an_epoch_with_nothing_to_report_is_refused():
    with pytest.raises(ValueError, match="needs results"):
        merge_results(None, None, None)


def test_train_cannot_be_added_to_a_results_value_that_is_not_a_dict():
    with pytest.raises(ValueError, match="not a dict"):
        merge_results({"train": 1}, {"loss": 1}, None)


def test_results_must_be_a_mapping():
    with pytest.raises(TypeError, match="must be a dict"):
        merge_results([1], None, None)


@dataclasses.dataclass
class Config:
    lr: float = 0.1


class Pydantic:
    def model_dump(self):
        return {"lr": 0.1}


@pytest.mark.parametrize(
    "config",
    [{"lr": 0.1}, argparse.Namespace(lr=0.1), Config(), Pydantic()],
    ids=["dict", "namespace", "dataclass", "pydantic"],
)
def test_a_config_is_read_in_the_shapes_it_comes_in(config):
    assert as_mapping(config) == {"lr": 0.1}


def test_a_config_nobody_can_read_is_refused():
    with pytest.raises(TypeError, match="cannot read a config"):
        as_mapping(object())


def test_an_unnamed_run_is_named_after_the_script(monkeypatch):
    monkeypatch.setattr("sys.argv", ["/jobs/train_unet.py"])
    assert default_name().startswith("train_unet ")
    monkeypatch.setattr("sys.argv", ["-c"])
    assert default_name().startswith("run ")
