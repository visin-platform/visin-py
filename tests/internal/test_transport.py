import base64
import hashlib

import pytest
import requests
from fakes import BASE, FakeResponse, FakeSession, FakeStream, ok, refused
from urllib3.exceptions import MaxRetryError, NewConnectionError, ProtocolError

from visin._internal.transport import HttpClient, file_sha256, never_sent, worth_retrying_later
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


def test_an_address_without_a_scheme_is_https_and_a_pasted_token_loses_its_newline():
    client = HttpClient("vision-api.visin.eu/", " vsn_live_abc\n", session=FakeSession())
    assert client.base_url == "https://vision-api.visin.eu/api"
    assert client.session.headers["Authorization"] == "Bearer vsn_live_abc"


# ---------------------------------------------------------------- chunked uploads


def stored(size):
    return FakeResponse(200, {"size": size, "complete": False})


def test_a_file_that_fits_one_piece_is_one_plain_request(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"0123456789")
    seen = []
    client.put_file_chunked(
        f"{BASE}/signed",
        str(weights),
        "application/octet-stream",
        chunk_bytes=10,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert uploads.puts[0]["bytes"] == b"0123456789"
    assert "Content-Range" not in uploads.puts[0]["headers"]
    assert seen == [(10, 10)]


def test_a_large_file_goes_up_in_ranges_each_naming_where_it_sits(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    data = bytes(range(25))
    weights.write_bytes(data)
    uploads.answers = [stored(10), stored(20), FakeResponse(200, {"size": 25, "complete": True})]
    seen = []
    client.put_file_chunked(
        f"{BASE}/signed",
        str(weights),
        "application/octet-stream",
        chunk_bytes=10,
        progress=lambda done, total: seen.append(done),
    )
    assert [put["headers"]["Content-Range"] for put in uploads.puts] == [
        "bytes 0-9/25",
        "bytes 10-19/25",
        "bytes 20-24/25",
    ]
    assert b"".join(put["bytes"] for put in uploads.puts) == data
    assert all(put["headers"]["Content-Type"] == "application/octet-stream" for put in uploads.puts)
    assert seen == [10, 20, 25]


def test_a_reply_that_names_no_size_means_the_piece_landed(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(25)))
    client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)
    assert len(uploads.puts) == 3


def test_a_server_that_holds_more_than_assumed_is_believed(client, uploads, tmp_path):
    # A piece whose answer was lost: the server has it, and says so with a 409.
    weights = tmp_path / "m.pth"
    data = bytes(range(25))
    weights.write_bytes(data)
    uploads.answers = [stored(10), FakeResponse(409, {"size": 20}), stored(25)]
    client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)
    assert [put["headers"]["Content-Range"] for put in uploads.puts] == [
        "bytes 0-9/25",
        "bytes 10-19/25",
        "bytes 20-24/25",
    ]


def test_a_piece_is_sent_again_after_a_dropped_connection_or_a_busy_server(client, uploads, sleeps, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(15)))
    uploads.answers = [dropped_mid_request(), FakeResponse(503, None, text="busy"), stored(10), stored(15)]
    client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)
    # The first piece is tried three times (dropped, busy, landed), the second once.
    assert [put["headers"]["Content-Range"] for put in uploads.puts] == [
        "bytes 0-9/15",
        "bytes 0-9/15",
        "bytes 0-9/15",
        "bytes 10-14/15",
    ]
    assert len(sleeps) == 2


def test_a_refusal_ends_the_upload_with_what_the_server_said(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(25)))
    uploads.answers = [stored(10), FakeResponse(403, None, text="expired")]
    with pytest.raises(ApiError, match="expired") as caught:
        client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)
    assert caught.value.status == 403


def test_an_upload_that_never_gets_past_a_piece_gives_up(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(25)))
    uploads.answers = [stored(0)] * 10
    with pytest.raises(TransportError, match="no progress"):
        client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)


def test_an_upload_whose_server_keeps_changing_its_mind_gives_up(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(25)))
    uploads.answers = [FakeResponse(409, {"size": size}) for size in range(1, 12)]
    with pytest.raises(TransportError, match="resynchronise"):
        client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)


def test_a_conflict_that_names_where_we_already_are_is_a_refusal_after_the_retries(client, uploads, tmp_path):
    # The server is still finishing an earlier attempt at this piece: worth a pause, and then it is an error.
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(25)))
    uploads.answers = [FakeResponse(409, {"size": 0})] * 10
    with pytest.raises(ApiError) as caught:
        client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)
    assert caught.value.status == 409


def test_a_connection_that_stays_down_ends_the_upload(client, uploads, tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(bytes(range(25)))
    uploads.answers = [connection_refused()] * 10
    with pytest.raises(TransportError, match="upload to signed URL failed"):
        client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)
    uploads.answers = [requests.exceptions.InvalidURL("bad")]
    with pytest.raises(TransportError, match="upload to signed URL failed"):
        client.put_file_chunked(f"{BASE}/signed", str(weights), "application/octet-stream", chunk_bytes=10)


# ---------------------------------------------------------------- authenticated, verified downloads


def test_an_api_download_is_made_as_the_client_and_accepts_any_type(client, session, tmp_path):
    session.route("GET", "/trainings/t1/models/m1/file", FakeStream(200, b"weights"))
    target = tmp_path / "m.pth"
    client.download_file(
        f"{client.base_url}/trainings/t1/models/m1/file", str(target), size=7, authenticated=True
    )
    assert target.read_bytes() == b"weights"
    call = session.calls[0]
    assert call["stream"] is True and call["headers"]["Accept"] == "*/*"


def test_a_download_with_the_wrong_digest_is_deleted_and_refused(client, session, tmp_path):
    session.route("GET", "/file", FakeStream(200, b"weights"), FakeStream(200, b"weights"))
    target = tmp_path / "m.pth"
    url = f"{client.base_url}/file"
    with pytest.raises(TransportError, match="SHA-256"):
        client.download_file(url, str(target), size=7, sha256="0" * 64, authenticated=True)
    assert not target.exists() and not (tmp_path / "m.pth.part").exists()

    good = hashlib.sha256(b"weights").hexdigest()
    client.download_file(url, str(target), size=7, sha256=good.upper(), authenticated=True)
    assert target.read_bytes() == b"weights"


def test_file_sha256_reads_a_file_in_pieces(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (9 * 2**20))
    assert file_sha256(str(big)) == hashlib.sha256(b"x" * (9 * 2**20)).hexdigest()


@pytest.mark.parametrize("partial", [False, True])
def test_repr_digest_verifies_whole_model_even_for_a_resumed_download(client, session, tmp_path, partial):
    data = b"checkpoint weights"
    digest = hashlib.sha256(data).digest()
    response = FakeStream(206 if partial else 200, data[5:] if partial else data)
    response.headers = {"Repr-Digest": "sha-256=:" + base64.b64encode(digest).decode() + ":"}
    session.route("GET", "/file", response)
    target = tmp_path / "model.pth"
    if partial:
        (tmp_path / "model.pth.part").write_bytes(data[:5])
    client.download_file(f"{client.base_url}/file", str(target), size=len(data), authenticated=True)
    assert target.read_bytes() == data


@pytest.mark.parametrize("header", ["sha-256=:bad!:", "sha-256=:YQ==:", "sha-256=wrong"])
def test_invalid_repr_digest_is_refused_before_writing_bytes(client, session, tmp_path, header):
    response = FakeStream(200, b"weights")
    response.headers = {"Repr-Digest": header}
    session.route("GET", "/file", response)
    target = tmp_path / "model.pth"
    with pytest.raises(TransportError, match="invalid Repr-Digest"):
        client.download_file(f"{client.base_url}/file", str(target), authenticated=True)
    assert not target.exists()


def test_repr_digest_must_agree_with_the_recorded_model_digest(client, session, tmp_path):
    response = FakeStream(200, b"weights")
    response.headers = {
        "Repr-Digest": "sha-256=:" + base64.b64encode(hashlib.sha256(b"weights").digest()).decode() + ":"
    }
    session.route("GET", "/file", response)
    with pytest.raises(TransportError, match="does not match"):
        client.download_file(
            f"{client.base_url}/file", str(tmp_path / "model.pth"), sha256="0" * 64, authenticated=True
        )


def test_upload_refuses_an_impossible_server_offset_and_nonpositive_chunk_size(client, uploads, tmp_path):
    path = tmp_path / "weights.pth"
    path.write_bytes(b"x" * 20)
    with pytest.raises(ValueError, match="positive"):
        client.put_file_chunked(f"{BASE}/signed", str(path), "application/octet-stream", chunk_bytes=0)
    uploads.answers = [stored(21)]
    with pytest.raises(TransportError, match="reports 21 bytes"):
        client.put_file_chunked(f"{BASE}/signed", str(path), "application/octet-stream", chunk_bytes=10)
