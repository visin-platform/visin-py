# Contributing

Thanks for helping. This page covers setting up, the checks a change has to pass, and how a release
happens.

## Set up

```sh
git clone https://github.com/visin-platform/visin-py
cd visin-py
make install          # .venv with every tool, plus the pre-commit hooks
```

`make help` lists every target. Each one runs the same command CI runs.

## Before you open a pull request

```sh
make check
```

This runs, in order:

| Target | What it checks |
| --- | --- |
| `make lint` | `ruff check` and `ruff format --check`. `make format` fixes most findings. |
| `make typecheck` | `mypy --strict` over the package. |
| `make coverage` | The whole test suite, failing if coverage drops below the floor in `pyproject.toml`. |
| `make build` | Builds the sdist and wheel, and checks their metadata with `twine check --strict`. |
| `make docs` | Builds the documentation site with `mkdocs build --strict`, so a broken link or reference fails. |

`make contract` checks every request the package sends against Visin's API spec. It reads the spec
from `VISIN_OPENAPI`, or from a Visin checkout next to this one at `../visin`. CI runs it against
Visin's main branch.

The coverage floor sits just below current coverage. Raise it when you add tests, and never lower it
to make a change pass.

## Where things are

The package's layout, and the reasoning behind its main decisions, are in
[How it is built](https://visin-platform.github.io/visin-py/architecture/). The tests mirror the package, and
[tests/README.md](https://github.com/visin-platform/visin-py/blob/main/tests/README.md) says what each file covers.

## Writing a change

- **Keep the loop safe.** Nothing in `Run` may raise into a training loop (starting a run aside) unless `strict=True` is
  set. Report problems through `_handle`.
- **New request fields** must exist in Visin's API. The contract test fails on a field the server
  would strip.
- **Support Python 3.9.** CI tests 3.9 to 3.14. `from __future__ import annotations` makes the
  newer annotation syntax safe. Runtime expressions such as `isinstance(x, int | str)` are not.
- **No new runtime dependencies** without a strong reason. The package goes into any training image,
  and `requests` is its only dependency. Optional features belong behind an extra.
- **Test what the change promises**, through the public API where you can, using the fakes in
  `tests/fakes.py`. No test may touch the network or `~/.visin`; `tests/conftest.py` enforces this.
- **Documentation:** a user-facing change updates the relevant page under `docs/`, and docstrings
  feed the API reference.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/), because the release
workflow builds the changelog and picks the version from them:

| Prefix | Changelog section | Version bump |
| --- | --- | --- |
| `feat:` | Added | minor |
| `fix:` | Fixed | patch |
| `perf:` | Changed | patch |
| `feat!:`, or a `BREAKING CHANGE:` footer | Breaking changes | major (minor before 1.0) |
| `docs:`, `test:`, `ci:`, `chore:`, `refactor:`, `build:`, `style:` | not listed | patch, when released |

For anything a user should know that a commit subject cannot say, add a line under `## [Unreleased]`
in `CHANGELOG.md`. The release keeps it.

## Releasing

A maintainer runs the **Release** workflow from the Actions tab, and that one run does everything:

1. runs the full CI suite;
2. works out the version from the commits since the last tag, or uses the one given
   (`patch`, `minor`, `major` or `1.4.0`);
3. writes `src/visin/_version.py` and the `CHANGELOG.md` entry, commits `chore(release): X.Y.Z`
   and tags `vX.Y.Z`;
4. builds, and publishes to PyPI through trusted publishing, so no API token is stored anywhere;
5. creates the GitHub release with the changelog entry and the built files attached;
6. publishes the documentation.

Choose `testpypi` as the target to rehearse this. That builds and publishes to TestPyPI, and commits,
tags and releases nothing.

`python scripts/release.py --dry-run` shows locally what the next release would be.

### One-time setup

- On PyPI and TestPyPI, add a trusted publisher for `visin-platform/visin-py`, workflow `release.yml`,
  with the environments `pypi` and `testpypi`.
- In the repository settings, create those two environments. Adding required reviewers to `pypi`
  makes every release wait for approval.
- Under **Pages**, set the source to **GitHub Actions**.
