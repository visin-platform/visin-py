from visin._internal.payloads import MAX_NAME, run_payload, tag_list, update_body


def test_a_lone_tag_is_one_tag():
    assert tag_list("ablation") == ["ablation"]
    assert tag_list(("a", "b")) == ["a", "b"]


def test_a_name_over_the_cap_is_cut_and_kept_whole_as_the_description():
    name = "x" * (MAX_NAME + 50)
    payload = run_payload("u", name)
    assert payload["name"] == "x" * MAX_NAME
    assert payload["description"] == name


def test_a_given_description_is_not_replaced_by_a_long_name():
    payload = run_payload("u", "x" * (MAX_NAME + 1), description="mine")
    assert payload["description"] == "mine"


def test_a_model_and_dataset_travel_under_metadata():
    payload = run_payload("u", "n", model="unet", dataset="ds1", metadata={"a": 1})
    assert payload["metadata"] == {"a": 1, "model": "unet", "dataset": "ds1"}
    assert payload["datasetId"] == "ds1"


def test_an_update_carries_only_what_was_given():
    assert update_body() == {}
    assert update_body(name="   ") == {}
    assert update_body(name=" new ", tags="t") == {"name": "new", "tags": ["t"]}
