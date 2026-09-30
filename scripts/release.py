#!/usr/bin/env python3
"""Cut a release: pick the version, write it, and write the changelog entry.

    python scripts/release.py                 # version from the commits since the last tag
    python scripts/release.py minor           # or patch, major, or an exact 1.4.0
    python scripts/release.py --dry-run       # show what would happen, change nothing
    python scripts/release.py --notes 1.4.0   # print that version's changelog entry

The version comes from Conventional Commits, as the monorepo's releases do:
a breaking change is major (minor before 1.0), a ``feat`` is minor, and
anything else is a patch. The entry lists ``feat``, ``fix`` and ``perf``
commits and breaking changes, and keeps whatever was written by hand under
``## [Unreleased]``.

It writes ``src/visin/_version.py`` and ``CHANGELOG.md`` and prints the new
version. It does not commit, tag or publish; the release workflow does those.
Standard library only, so it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "visin" / "_version.py"
CHANGELOG = ROOT / "CHANGELOG.md"
REPO_URL = "https://github.com/visin-platform/visin-py"

_SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<subject>.+)$")
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
SECTIONS = {"feat": "Added", "fix": "Fixed", "perf": "Changed"}
SECTION_ORDER = ["Breaking changes", "Added", "Changed", "Fixed"]


@dataclass(frozen=True)
class Commit:
    type: str
    scope: str | None
    subject: str
    breaking: bool
    breaking_note: str | None = None


def parse_commit(message: str) -> Commit | None:
    """A Conventional Commit, or None for a message that is not one."""
    subject, _, body = message.strip().partition("\n")
    match = _SUBJECT.match(subject.strip())
    if not match:
        return None
    note = None
    footer = re.search(r"^BREAKING[ -]CHANGE: (.+)$", body, re.MULTILINE)
    if footer:
        note = footer.group(1).strip()
    return Commit(
        type=match["type"],
        scope=match["scope"],
        subject=match["subject"].strip(),
        breaking=bool(match["bang"] or footer),
        breaking_note=note,
    )


def parse_version(text: str) -> tuple[int, int, int]:
    match = _VERSION.match(text.strip().lstrip("v"))
    if not match:
        raise ValueError(f"not a version: {text!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def next_version(current: str, commits: list[Commit], release_as: str | None) -> str:
    """The version to release, from an explicit request or from the commits."""
    major, minor, patch = parse_version(current)
    if release_as and _VERSION.match(release_as.lstrip("v")):
        return format_version(parse_version(release_as))
    bump = release_as
    if not bump:
        if any(commit.breaking for commit in commits):
            bump = "major"
        elif any(commit.type == "feat" for commit in commits):
            bump = "minor"
        else:
            bump = "patch"
    if bump == "major":
        # Before 1.0 a breaking change is a minor: 0.x promises no stability.
        return format_version((major + 1, 0, 0) if major >= 1 else (0, minor + 1, 0))
    if bump == "minor":
        return format_version((major, minor + 1, 0))
    if bump == "patch":
        return format_version((major, minor, patch + 1))
    raise ValueError(f"release-as must be patch, minor, major or a version, not {bump!r}")


def render_entry(commits: list[Commit]) -> dict[str, list[str]]:
    """Changelog lines from commits, by section."""
    sections: dict[str, list[str]] = {}
    for commit in commits:
        scope = f"**{commit.scope}:** " if commit.scope else ""
        if commit.breaking:
            sections.setdefault("Breaking changes", []).append(
                f"- {scope}{commit.breaking_note or commit.subject}"
            )
        section = SECTIONS.get(commit.type)
        if section:
            sections.setdefault(section, []).append(f"- {scope}{commit.subject}")
    return sections


def _split_unreleased(text: str) -> tuple[str, str, str]:
    """The changelog as (before Unreleased, its body, the rest)."""
    marker = "## [Unreleased]"
    start = text.index(marker)
    body_start = start + len(marker)
    following = re.search(r"^## \[", text[body_start:], re.MULTILINE)
    links = re.search(r"^\[[^\]]+\]: ", text[body_start:], re.MULTILINE)
    ends = [m.start() for m in (following, links) if m]
    body_end = body_start + (min(ends) if ends else len(text) - body_start)
    return text[:start], text[body_start:body_end], text[body_end:]


def _merge(hand_written: str, generated: dict[str, list[str]]) -> str:
    """Hand-written notes first in each section, then the generated lines."""
    sections: dict[str, list[str]] = {}
    current = None
    loose: list[str] = []
    for line in hand_written.strip().splitlines():
        heading = re.match(r"^### (.+)$", line)
        if heading:
            current = heading.group(1).strip()
            sections.setdefault(current, [])
        elif line.strip():
            (sections[current] if current else loose).append(line)
        elif current and sections[current]:
            sections[current].append(line)
    for name, lines in generated.items():
        existing = sections.setdefault(name, [])
        for line in lines:
            if line not in existing:
                existing.append(line)
    order = SECTION_ORDER + [name for name in sections if name not in SECTION_ORDER]
    parts = ["\n".join(loose)] if loose else []
    for name in order:
        lines = list(sections.get(name, []))
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            parts.append(f"### {name}\n\n" + "\n".join(lines))
    return "\n\n".join(parts)


def update_changelog(
    text: str, version: str, previous: str | None, generated: dict[str, list[str]], date: str
) -> str:
    before, unreleased, rest = _split_unreleased(text)
    body = _merge(unreleased, generated) or "No user-facing changes."
    entry = f"## [Unreleased]\n\n## [{version}] - {date}\n\n{body}\n\n"
    # Compare links: Unreleased against the new tag, the new version against the one before.
    rest = re.sub(r"^\[Unreleased\]: .*\n?", "", rest, flags=re.MULTILINE)
    links = f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD\n"
    if previous:
        links += f"[{version}]: {REPO_URL}/compare/v{previous}...v{version}\n"
    else:
        links += f"[{version}]: {REPO_URL}/releases/tag/v{version}\n"
    rest = rest.strip("\n")
    return before + entry + links + (rest + "\n" if rest else "")


def release_notes(text: str, version: str) -> str:
    """One version's entry, for the GitHub release."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"no changelog entry for {version}")
    return match.group(1).strip() + "\n"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def last_tag() -> str | None:
    try:
        return _git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*").strip() or None
    except subprocess.CalledProcessError:
        return None


def commits_since(tag: str | None) -> list[Commit]:
    try:
        _git("rev-parse", "--verify", "HEAD")
    except subprocess.CalledProcessError:
        return []  # a repository with no commits yet
    log = _git("log", "--format=%B%x1e", f"{tag}..HEAD" if tag else "HEAD")
    parsed = (parse_commit(message) for message in log.split("\x1e") if message.strip())
    return [commit for commit in parsed if commit and not commit.subject.startswith("release")]


def read_version() -> str:
    match = re.search(r'__version__ = "([^"]+)"', VERSION_FILE.read_text())
    if not match:
        raise SystemExit(f"no __version__ in {VERSION_FILE}")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("release_as", nargs="?", help="patch, minor, major or an exact version")
    parser.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    parser.add_argument("--notes", metavar="VERSION", help="print a version's changelog entry and exit")
    args = parser.parse_args(argv)

    if args.notes:
        sys.stdout.write(release_notes(CHANGELOG.read_text(), args.notes.lstrip("v")))
        return 0

    tag = last_tag()
    previous = tag.lstrip("v") if tag else None
    commits = commits_since(tag)
    current = read_version()
    if previous is None and not args.release_as:
        version = current  # the first release is the version already written
    else:
        version = next_version(previous or current, commits, args.release_as)
    if previous and parse_version(version) <= parse_version(previous):
        raise SystemExit(f"{version} is not after the last release, {previous}")

    generated = render_entry(commits)
    date = datetime.date.today().isoformat()
    changelog = update_changelog(CHANGELOG.read_text(), version, previous, generated, date)
    if args.dry_run:
        print(f"would release {version} (last release: {previous or 'none'}; {len(commits)} commits)\n")
        print(release_notes(changelog, version))
        return 0
    VERSION_FILE.write_text(f'__version__ = "{version}"\n')
    CHANGELOG.write_text(changelog)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
