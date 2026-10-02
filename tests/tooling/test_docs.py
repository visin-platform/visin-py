"""The documentation names only what the package has.

Cheap checks that catch the usual drift: an environment variable that is no
longer read, a command that was renamed, a public name that was removed.
"""

import re
from pathlib import Path

import pytest

import visin
from visin._internal import config
from visin.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
PAGES = [page for page in PAGES if page.name not in ("changelog.md", "contributing.md", "architecture.md")]


def text(page):
    return page.read_text(encoding="utf-8")


def known_variables():
    names = {
        value for key, value in vars(config).items() if key.startswith("ENV_") and isinstance(value, str)
    }
    names |= {
        name
        for key, value in vars(config).items()
        if key.startswith("ENV_") and isinstance(value, tuple)
        for name in value
    }
    return names


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_variable_a_page_names_is_one_the_package_reads(page):
    named = set(re.findall(r"\bVISIN_[A-Z_]+\b", text(page)))
    unknown = named - known_variables() - {"VISIN_REQUIRE_CONTRACT", "VISIN_OPENAPI"}
    assert not unknown, f"{page.name} names variables the package does not read: {sorted(unknown)}"


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_visin_command_a_page_shows_exists(page):
    commands = set(build_parser()._subparsers._group_actions[0].choices)
    shown = set(re.findall(r"^(?:\$ |\s*)visin(?: -v)? ([a-z]+)\b", text(page), flags=re.MULTILINE))
    shown |= set(re.findall(r"`visin(?: -v)? ([a-z]+)", text(page)))
    unknown = {name for name in shown if name not in commands}
    assert not unknown, f"{page.name} shows commands that do not exist: {sorted(unknown)}"


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_visin_name_a_page_calls_exists(page):
    called = set(re.findall(r"\bvisin\.([A-Za-z_]+)\(", text(page)))
    missing = {name for name in called if not hasattr(visin, name)}
    assert not missing, f"{page.name} calls names visin does not have: {sorted(missing)}"
