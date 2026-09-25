import threading
import time

from visin._internal.sender import Sender


def test_work_runs_on_the_background_thread():
    seen = []
    sender = Sender()
    sender.submit(lambda: seen.append(threading.current_thread().name))
    assert sender.flush(timeout=5)
    assert seen and seen[0] == "visin-sender"


def test_ordering_is_preserved():
    seen = []
    sender = Sender()
    for index in range(20):
        sender.submit(lambda index=index: seen.append(index))
    assert sender.flush(timeout=5)
    assert seen == list(range(20))


def test_a_failing_report_does_not_stop_the_thread():
    seen = []

    def boom():
        raise RuntimeError("upstream down")

    sender = Sender()
    sender.submit(boom)
    sender.submit(lambda: seen.append("after"))
    assert sender.flush(timeout=5)
    assert seen == ["after"]
    assert sender.failed == 1


def test_a_full_queue_drops_rather_than_blocking():
    sender = Sender(max_queue=1)
    blocked = threading.Event()
    sender.submit(blocked.wait)  # occupies the worker
    for _ in range(50):
        sender.submit(lambda: None)
    assert sender.dropped > 0
    blocked.set()
    sender.stop(timeout=5)


def test_flush_returns_true_when_nothing_was_ever_queued():
    assert Sender().flush(timeout=1) is True


def test_flush_after_stop_returns_at_once():
    sender = Sender()
    sender.submit(lambda: None)
    sender.stop(timeout=5)
    started = time.monotonic()
    assert sender.flush(timeout=5) is True
    assert time.monotonic() - started < 1


def test_a_stop_that_cannot_wait_hands_queued_work_back():
    blocked = threading.Event()
    sender = Sender()
    sender.submit(blocked.wait)  # in flight, and stuck
    queued = [lambda: None, lambda: None]
    for work in queued:
        assert sender.submit(work)
    assert sender.stop(timeout=0.1) is False
    assert sender.leftover == queued
    blocked.set()
    assert sender.submit(lambda: None) is False  # stopped


def test_submit_says_when_it_dropped_work():
    blocked = threading.Event()
    sender = Sender(max_queue=1)
    sender.submit(blocked.wait)
    results = [sender.submit(lambda: None) for _ in range(5)]
    assert False in results
    blocked.set()
    sender.stop(timeout=5)
