"""The Keras and Lightning callbacks, driven the way each framework calls them.

Neither framework is a dependency, so each test stands in a minimal module that
provides the ``Callback`` base class the integration subclasses.
"""

import importlib
import sys
import types

import pytest

from visin.integrations._metrics import split_metrics


class RecordingRun:
    """What a callback calls on a run, recorded."""

    enabled = True

    def __init__(self):
        self.epochs = []
        self.tests = []
        self.configs = []
        self.finished = None
        self.last_epoch = None

    def log_epoch(self, epoch, results=None, **kwargs):
        self.epochs.append({"epoch": epoch, **kwargs})
        self.last_epoch = epoch

    def log_test_results(self, epoch, results):
        self.tests.append((epoch, results))

    def log_config(self, config, name=None):
        self.configs.append((config, name))

    def finish(self):
        self.finished = "completed"

    def fail(self, error=None):
        self.finished = "failed"


class BaseCallback:
    def __init__(self):
        self.model = None


def fresh_import(monkeypatch, name, fake_modules):
    for module_name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def keras_integration(monkeypatch):
    keras = types.ModuleType("keras")
    keras.callbacks = types.SimpleNamespace(Callback=BaseCallback)
    return fresh_import(monkeypatch, "visin.integrations.keras", {"keras": keras})


@pytest.fixture
def lightning_integration(monkeypatch):
    callbacks = types.ModuleType("lightning.pytorch.callbacks")
    callbacks.Callback = BaseCallback
    return fresh_import(
        monkeypatch,
        "visin.integrations.lightning",
        {
            "lightning": types.ModuleType("lightning"),
            "lightning.pytorch": types.ModuleType("lightning.pytorch"),
            "lightning.pytorch.callbacks": callbacks,
        },
    )


@pytest.fixture
def huggingface_integration(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = BaseCallback
    return fresh_import(monkeypatch, "visin.integrations.huggingface", {"transformers": transformers})


# ---------------------------------------------------------------- metric names


def test_prefixes_become_curves():
    assert split_metrics({"loss": 1, "val_loss": 2, "val/iou": 3, "train_acc": 4, "test_iou": 5}) == {
        "train": {"loss": 1, "acc": 4},
        "val": {"loss": 2, "iou": 3},
        "test": {"iou": 5},
    }


def test_lightning_step_copies_are_dropped_and_epoch_copies_renamed():
    assert split_metrics({"train_loss_step": 1, "train_loss_epoch": 2, "lr": 0.1}) == {"train": {"loss": 2}}


def test_keras_mean_io_u_becomes_the_mean_iou_the_charts_read():
    assert split_metrics({"val_mean_io_u": 0.5}) == {"val": {"mean_iou": 0.5}}


def test_a_bare_prefix_is_not_a_metric():
    assert split_metrics({"val_": 1}) == {}


# ---------------------------------------------------------------- Keras


def test_keras_reports_each_epoch_numbered_as_keras_prints_it(keras_integration):
    run = RecordingRun()
    callback = keras_integration.VisinCallback(run)
    callback.model = types.SimpleNamespace(
        optimizer=types.SimpleNamespace(learning_rate=0.01, iterations=0),
    )
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0, {"loss": 0.9, "val_loss": 1.1, "val_mean_io_u": 0.4})
    callback.on_epoch_end(1, {})  # nothing measured, nothing sent
    callback.on_train_end()
    (epoch,) = run.epochs
    assert epoch["epoch"] == 1
    assert epoch["train"] == {"loss": 0.9}
    assert epoch["val"] == {"loss": 1.1, "mean_iou": 0.4}
    assert epoch["learning_rate"] == 0.01
    assert epoch["epoch_time"] >= 0
    assert run.finished is None  # a run it was given is the caller's to finish


def test_keras_prefers_the_logged_learning_rate(keras_integration):
    run = RecordingRun()
    callback = keras_integration.VisinCallback(run)
    callback.on_epoch_end(4, {"loss": 1, "learning_rate": 0.5})
    assert run.epochs[0]["learning_rate"] == 0.5 and run.epochs[0]["epoch"] == 5


def test_keras_starts_and_finishes_a_run_of_its_own(keras_integration, monkeypatch):
    run = RecordingRun()
    started = {}
    monkeypatch.setattr(keras_integration, "init", lambda **kwargs: started.update(kwargs) or run)
    callback = keras_integration.VisinCallback(name="unet", project="road-seg")
    callback.on_train_begin()
    callback.on_epoch_end(0, {"loss": 1})
    callback.on_train_end()
    assert started == {"name": "unet", "project": "road-seg"}
    assert run.finished == "completed"


def test_keras_learning_rate_schedules_are_evaluated(keras_integration):
    schedule = lambda step: 0.25  # noqa: E731
    model = types.SimpleNamespace(optimizer=types.SimpleNamespace(learning_rate=schedule, iterations=3))
    assert keras_integration._learning_rate(model) == 0.25
    assert keras_integration._learning_rate(object()) is None


# ---------------------------------------------------------------- Lightning


def trainer(**overrides):
    values = {
        "is_global_zero": True,
        "sanity_checking": False,
        "current_epoch": 0,
        "callback_metrics": {},
        "optimizers": [types.SimpleNamespace(param_groups=[{"lr": 0.001}])],
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_lightning_reports_each_epoch_with_its_validation(lightning_integration, monkeypatch):
    run = RecordingRun()
    monkeypatch.setattr(lightning_integration, "init", lambda **kwargs: run)
    callback = lightning_integration.VisinCallback(name="segformer")
    module = types.SimpleNamespace(hparams={"lr": 0.001, "backbone": "b2"})
    t = trainer(
        callback_metrics={"train_loss_epoch": 0.8, "train_loss_step": 0.7, "val_loss": 0.9, "val/iou": 0.5}
    )
    callback.setup(t, module, "fit")
    callback.on_train_epoch_start(t, module)
    callback.on_validation_end(t, module)
    callback.on_train_epoch_end(t, module)
    callback.on_fit_end(t, module)
    assert run.configs == [({"lr": 0.001, "backbone": "b2"}, "SimpleNamespace")]
    (epoch,) = run.epochs
    assert epoch["epoch"] == 1
    assert epoch["train"] == {"loss": 0.8}
    assert epoch["val"] == {"loss": 0.9, "iou": 0.5}
    assert epoch["learning_rate"] == 0.001
    assert run.finished == "completed"


def test_lightning_skips_sanity_checks_and_other_ranks(lightning_integration):
    run = RecordingRun()
    callback = lightning_integration.VisinCallback(run)
    callback.on_train_epoch_end(trainer(sanity_checking=True, callback_metrics={"val_loss": 1}), None)
    callback.on_train_epoch_end(trainer(is_global_zero=False, callback_metrics={"val_loss": 1}), None)
    callback.on_train_epoch_end(trainer(callback_metrics={}), None)
    assert run.epochs == []


def test_lightning_does_not_start_a_run_on_other_ranks(lightning_integration, monkeypatch):
    monkeypatch.setattr(lightning_integration, "init", lambda **kwargs: pytest.fail("started a run"))
    callback = lightning_integration.VisinCallback()
    callback.setup(trainer(is_global_zero=False), None, "fit")
    assert callback.run is None


def test_lightning_test_metrics_become_a_test_result_on_the_last_epoch(lightning_integration):
    run = RecordingRun()
    callback = lightning_integration.VisinCallback(run, finish_after="test")
    callback.on_train_epoch_end(trainer(current_epoch=9, callback_metrics={"train_loss": 1}), None)
    callback.on_test_end(trainer(callback_metrics={"test_iou": 0.7, "test_loss": 0.3}), None)
    assert run.tests == [(10, {"overall": {"iou": 0.7, "loss": 0.3}})]


def test_lightning_test_without_an_epoch_is_not_sent(lightning_integration):
    run = RecordingRun()
    lightning_integration.VisinCallback(run).on_test_end(trainer(callback_metrics={"test_iou": 0.7}), None)
    assert run.tests == []


def test_lightning_marks_its_run_failed_on_an_exception(lightning_integration, monkeypatch):
    run = RecordingRun()
    monkeypatch.setattr(lightning_integration, "init", lambda **kwargs: run)
    callback = lightning_integration.VisinCallback(log_hyperparameters=False)
    callback.setup(trainer(), types.SimpleNamespace(), "fit")
    callback.on_exception(trainer(), None, RuntimeError("OOM"))
    assert run.finished == "failed"


def test_lightning_rejects_an_unknown_finish_point(lightning_integration):
    with pytest.raises(ValueError):
        lightning_integration.VisinCallback(finish_after="sometime")


def test_lightning_without_an_optimizer_reports_no_learning_rate(lightning_integration):
    assert lightning_integration._learning_rate(trainer(optimizers=[])) is None


def test_a_missing_framework_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightning.pytorch.callbacks", None)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", None)
    monkeypatch.delitem(sys.modules, "visin.integrations.lightning", raising=False)
    with pytest.raises(ImportError, match="pip install lightning"):
        importlib.import_module("visin.integrations.lightning")


# ---------------------------------------------------------------- Hugging Face


def trainer_state(epoch):
    return types.SimpleNamespace(epoch=epoch, is_world_process_zero=True)


def drive(callback, epochs, evaluate_after_epoch_end=True):
    """The events a Trainer raises, in order: an end-of-epoch evaluation comes after the epoch-end event."""
    args = types.SimpleNamespace(to_dict=lambda: {"lr": 1e-3})
    callback.on_train_begin(args, trainer_state(0), None)
    for number in range(1, epochs + 1):
        state = trainer_state(float(number))
        callback.on_epoch_begin(args, trainer_state(number - 1.0), None)
        callback.on_log(
            args, state, None, {"loss": 1.0 / number, "learning_rate": 0.1, "grad_norm": 2.0, "epoch": number}
        )
        callback.on_epoch_end(args, state, None)
        if evaluate_after_epoch_end:
            callback.on_evaluate(
                args,
                state,
                None,
                {"eval_loss": 1.5 / number, "eval_mean_iou": 0.1 * number, "eval_runtime": 3.0},
            )
    callback.on_train_end(args, trainer_state(float(epochs)), None)


def test_huggingface_attaches_an_end_of_epoch_evaluation_to_its_epoch(huggingface_integration):
    run = RecordingRun()
    drive(huggingface_integration.VisinCallback(run), 2)
    assert [e["epoch"] for e in run.epochs] == [1, 2]
    first = run.epochs[0]
    assert first["train"] == {"loss": 1.0, "grad_norm": 2.0}
    assert first["val"] == {"loss": 1.5, "mean_iou": 0.1}
    assert first["learning_rate"] == 0.1 and first["epoch_time"] >= 0
    assert run.finished is None


def test_huggingface_sends_the_last_epoch_when_training_ends(huggingface_integration):
    run = RecordingRun()
    drive(huggingface_integration.VisinCallback(run), 1, evaluate_after_epoch_end=False)
    assert [e["epoch"] for e in run.epochs] == [1]
    assert run.epochs[0]["val"] is None


def test_huggingface_ignores_runtimes_and_throughput(huggingface_integration):
    run = RecordingRun()
    callback = huggingface_integration.VisinCallback(run)
    args = types.SimpleNamespace()
    callback.on_log(args, trainer_state(1.0), None, {"train_runtime": 9.0, "train_samples_per_second": 3.0})
    callback.on_epoch_end(args, trainer_state(1.0), None)
    callback.on_train_end(args, trainer_state(1.0), None)
    assert run.epochs == []


def test_huggingface_predict_metrics_become_a_test_result_on_the_last_epoch(huggingface_integration):
    run = RecordingRun()
    callback = huggingface_integration.VisinCallback(run)
    drive(callback, 2)
    callback.on_predict(None, trainer_state(2.0), None, {"test_loss": 0.3, "test_runtime": 1.0})
    assert run.tests == [(2, {"overall": {"loss": 0.3}})]


def test_huggingface_starts_logs_the_config_of_and_finishes_a_run_of_its_own(
    huggingface_integration, monkeypatch
):
    run = RecordingRun()
    started = {}
    monkeypatch.setattr(huggingface_integration, "init", lambda **kwargs: started.update(kwargs) or run)
    drive(huggingface_integration.VisinCallback(name="segformer"), 1)
    assert started == {"name": "segformer"}
    assert run.configs == [({"lr": 1e-3}, None)]
    assert run.finished == "completed"


def test_huggingface_only_reports_from_the_main_process(huggingface_integration, monkeypatch):
    started = []
    monkeypatch.setattr(huggingface_integration, "init", lambda **kwargs: started.append(1))
    huggingface_integration.VisinCallback().on_train_begin(
        None, types.SimpleNamespace(is_world_process_zero=False), None
    )
    assert started == []


def test_lightning_does_not_report_a_stale_validation_on_an_epoch_that_did_not_validate(
    lightning_integration,
):
    run = RecordingRun()
    callback = lightning_integration.VisinCallback(run)
    metrics = {"train_loss_epoch": 0.8, "val_loss": 0.9}
    callback.on_validation_end(trainer(current_epoch=0, callback_metrics=metrics), None)
    callback.on_train_epoch_end(trainer(current_epoch=0, callback_metrics=metrics), None)
    callback.on_train_epoch_end(trainer(current_epoch=1, callback_metrics=metrics), None)
    assert [(e["epoch"], e["val"]) for e in run.epochs] == [(1, {"loss": 0.9}), (2, None)]


def test_lightning_ignores_the_sanity_check_validation(lightning_integration):
    run = RecordingRun()
    callback = lightning_integration.VisinCallback(run)
    callback.on_validation_end(trainer(sanity_checking=True), None)
    callback.on_train_epoch_end(trainer(callback_metrics={"train_loss": 1, "val_loss": 2}), None)
    assert run.epochs[0]["val"] is None


def test_lightning_test_alone_does_not_start_a_run(lightning_integration, monkeypatch):
    monkeypatch.setattr(lightning_integration, "init", lambda **kwargs: pytest.fail("started a run"))
    callback = lightning_integration.VisinCallback(name="segformer")
    callback.setup(trainer(), types.SimpleNamespace(), "test")
    callback.setup(trainer(), types.SimpleNamespace(), "validate")
    assert callback.run is None


def test_lightning_a_second_fit_starts_a_new_run(lightning_integration, monkeypatch):
    runs = [RecordingRun(), RecordingRun()]
    monkeypatch.setattr(lightning_integration, "init", lambda **kwargs: runs.pop(0))
    callback = lightning_integration.VisinCallback(log_hyperparameters=False)
    module = types.SimpleNamespace()
    callback.setup(trainer(), module, "fit")
    first = callback.run
    callback.on_fit_end(trainer(), module)
    callback.setup(trainer(), module, "test")
    assert callback.run is first
    callback.setup(trainer(), module, "fit")
    assert callback.run is not first and first.finished == "completed"


def test_keras_a_second_fit_starts_a_new_run(keras_integration, monkeypatch):
    runs = [RecordingRun(), RecordingRun()]
    monkeypatch.setattr(keras_integration, "init", lambda **kwargs: runs.pop(0))
    callback = keras_integration.VisinCallback(name="unet")
    for _ in range(2):
        callback.on_train_begin()
        callback.on_epoch_end(0, {"loss": 1})
        callback.on_train_end()
    assert runs == [] and callback.run is None


def test_keras_initial_epoch_carries_on_the_numbering(keras_integration):
    run = RecordingRun()
    callback = keras_integration.VisinCallback(run)
    callback.on_epoch_end(9, {"loss": 1})
    assert run.epochs[0]["epoch"] == 10


def test_huggingface_a_second_train_starts_a_new_run(huggingface_integration, monkeypatch):
    runs = [RecordingRun(), RecordingRun()]
    monkeypatch.setattr(huggingface_integration, "init", lambda **kwargs: runs.pop(0))
    callback = huggingface_integration.VisinCallback(name="segformer")
    drive(callback, 1)
    drive(callback, 1)
    assert runs == [] and callback.run is None
