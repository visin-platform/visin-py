from fakes import ok

import visin
from visin import Run


def test_without_a_run_the_shortcuts_do_nothing():
    assert visin.get_run().mode == "disabled"
    assert visin.log_epoch(1, train={"loss": 1.0}) is None
    visin.finish()


def test_the_shortcuts_log_to_the_run_init_made(server, session):
    run = visin.init("shared")
    assert visin.get_run() is run
    uuid = visin.log_epoch(1, train={"loss": 1.0})
    visin.log_config({"lr": 1})
    visin.update(tags=["x"])
    visin.finish()
    assert uuid == run.epoch_uuid(1)
    assert session.paths("POST").count("/epochs/upload") == 1
    assert "/configs/upload" in session.paths("POST")


def test_attach_becomes_the_current_run(server, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    attached = Run.attach("existing", mark_status=False)
    assert visin.get_run() is attached


def test_a_run_made_by_create_is_not_made_current(server):
    Run.create("private")
    assert visin.get_run().mode == "disabled"
