import pytest
import requests
from fakes import BASE, FakeResponse, ok, refused
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProtocolError

from visin._internal.transport import HttpClient, never_sent, worth_retrying_later
from visin.errors import ApiError, TransportError


def connection_refused():
    reason = NewConnectionError(None, "Connection refused")
    return requests.exceptions.ConnectionError(MaxRetryError(None, "/", reason=reason))


def dropped_mid_request():
    return requests.exceptions.ConnectionError(ProtocolError("Connection aborted.", ConnectionResetError()))


# ---------------------------------------------------------------- basics


def test_base_url_tolerates_both_spellings(session):
    assert HttpClient("https://h.test", session=session).base_url == "https://h.test/api"
    assert HttpClient("https://h.test/api/", session=session).base_url == "https://h.test/api"


def test_envelope_is_unwrapped(client, session):
    session.route("GET", "/thing", ok({"_id": "abc"}))
    assert client.request("GET", "/thing") == {"_id": "abc"}


def test_a_bare_body_is_passed_through(client, session):
    session.route("GET", "/thing", FakeResponse(200, [1, 2]))
    assert client.request("GET", "/thing") == [1, 2]


def test_unsuccessful_envelope_raises_with_server_message(client, session):
    session.route("GET", "/thing", FakeResponse(200, {"success": False, "message": "nope"}))
    with pytest.raises(ApiError, match="nope"):
        client.request("GET", "/thing")


def test_an_error_names_the_servers_message(client, session):
    session.route("POST", "/epochs/upload", refused(404, "Training not found with uuid: abc"))
    with pytest.raises(ApiError) as caught:
        client.request("POST", "/epochs/upload", json={})
    assert caught.value.status == 404
    assert "Training not found with uuid: abc" in str(caught.value)


def test_a_plain_text_error_keeps_status_and_body(client, session):
    session.route("POST", "/thing", FakeResponse(400, None, text="name too long"))
    with pytest.raises(ApiError) as caught:
        client.request("POST", "/thing", json={})
    assert caught.value.status == 400
    assert "name too long" in caught.value.body


def test_non_json_success_is_an_error(client, session):
    session.route("GET", "/thing", FakeResponse(200, None, text="<html>"))
    with pytest.raises(ApiError, match="non-JSON"):
        client.request("GET", "/thing")


def test_token_is_sent_as_bearer(session):
    HttpClient("https://h.test", "secret", session=session)
    assert session.headers["Authorization"] == "Bearer secret"


def test_no_token_sends_no_auth_header(session):
    HttpClient("https://h.test", None, session=session)
    assert "Authorization" not in session.headers


def test_the_signed_upload_session_never_carries_the_token(client, uploads):
    assert "Authorization" not in uploads.headers


def test_nan_is_sent_as_null_so_the_body_stays_valid_json(client, session):
    client.request("POST", "/thing", json={"loss": float("nan")})
    assert session.calls[0]["json"] == {"loss": None}


# ---------------------------------------------------------------- retries by status


def test_a_read_is_retried_through_a_503(client, session, sleeps):
    session.route("GET", "/thing", refused(503), ok({"n": 1}))
    assert client.request("GET", "/thing") == {"n": 1}
    assert len(sleeps) == 1


def test_a_500_is_never_retried(client, session, sleeps):
    session.route("GET", "/thing", refused(500), ok())
    with pytest.raises(ApiError):
        client.request("GET", "/thing")
    assert sleeps == []


def test_a_plain_post_is_not_repeated_after_a_502(client, session):
    # The gateway may have passed it on before giving up; a repeat could store it twice.
    session.route("POST", "/benchmarks/upload", refused(502), ok())
    with pytest.raises(ApiError):
        client.request("POST", "/benchmarks/upload", json={})
    assert len(session.calls) == 1


def test_a_plain_post_is_repeated_after_a_429(client, session):
    # The rate limiter turns a request away before any handler sees it.
    session.route("POST", "/benchmarks/upload", FakeResponse(429, None, text="Too many requests"), ok())
    client.request("POST", "/benchmarks/upload", json={})
    assert len(session.calls) == 2


def test_an_idempotent_post_is_repeated_after_a_502(client, session):
    session.route("POST", "/epochs/upload", refused(502), ok())
    client.request("POST", "/epochs/upload", json={}, idempotent=True)
    assert len(session.calls) == 2


def test_retry_after_is_honoured(client, session, sleeps):
    session.route(
        "GET", "/thing", FakeResponse(429, None, text="slow down", headers={"Retry-After": "7"}), ok()
    )
    client.request("GET", "/thing")
    assert sleeps == [7.0]


def test_retries_give_up_with_the_last_answer(client, session, sleeps):
    session.route("GET", "/thing", *[refused(503, "db down")] * 10)
    with pytest.raises(ApiError, match="db down") as caught:
        client.request("GET", "/thing")
    assert caught.value.status == 503
    assert len(session.calls) == client.retries + 1


# ---------------------------------------------------------------- retries by transport failure


def test_a_connection_that_never_opened_is_safe_to_retry_for_any_request(client, session):
    session.route("POST", "/benchmarks/upload", connection_refused(), ok())
    client.request("POST", "/benchmarks/upload", json={})
    assert len(session.calls) == 2


def test_a_plain_post_that_dropped_mid_request_is_not_repeated(client, session):
    session.route("POST", "/benchmarks/upload", dropped_mid_request(), ok())
    with pytest.raises(TransportError):
        client.request("POST", "/benchmarks/upload", json={})
    assert len(session.calls) == 1


def test_an_idempotent_post_that_dropped_mid_request_is_repeated(client, session):
    session.route("POST", "/epochs/upload", dropped_mid_request(), ok())
    client.request("POST", "/epochs/upload", json={}, idempotent=True)
    assert len(session.calls) == 2


def test_a_read_timeout_is_retried_for_a_read(client, session):
    session.route("GET", "/thing", requests.exceptions.ReadTimeout("slow"), ok())
    client.request("GET", "/thing")
    assert len(session.calls) == 2


def test_a_tls_failure_is_never_retried(client, session):
    session.route("GET", "/thing", requests.exceptions.SSLError("bad cert"), ok())
    with pytest.raises(TransportError):
        client.request("GET", "/thing")
    assert len(session.calls) == 1


def test_never_sent_tells_connect_failures_from_dropped_requests():
    assert never_sent(connection_refused())
    assert never_sent(requests.exceptions.ConnectTimeout("t"))
    assert not never_sent(dropped_mid_request())
    assert not never_sent(requests.exceptions.ReadTimeout("t"))
    assert not never_sent(requests.exceptions.SSLError("t"))


def test_which_failures_are_worth_keeping_for_later():
    assert worth_retrying_later(TransportError("down"))
    assert worth_retrying_later(ApiError("x", status=503))
    assert not worth_retrying_later(ApiError("x", status=400))
    assert not worth_retrying_later(ApiError("x", status=500))


# ---------------------------------------------------------------- uploads


def test_put_file_sends_the_bytes_with_its_type(client, uploads, tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"\x89PNG data")
    client.put_file(f"{BASE}/signed", str(frame), "image/png")
    assert uploads.puts[0]["bytes"] == b"\x89PNG data"
    assert uploads.puts[0]["headers"] == {"Content-Type": "image/png"}


def test_put_file_retries_and_resends_the_whole_file(client, uploads, tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"abc")
    uploads.answers = [FakeResponse(503, None, text="busy"), FakeResponse(200, None)]
    client.put_file(f"{BASE}/signed", str(frame), "image/png")
    assert [put["bytes"] for put in uploads.puts] == [b"abc", b"abc"]


def test_put_file_refusal_is_an_api_error(client, uploads, tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"abc")
    uploads.answers = [FakeResponse(403, None, text="expired")]
    with pytest.raises(ApiError, match="expired"):
        client.put_file(f"{BASE}/signed", str(frame), "image/png")
