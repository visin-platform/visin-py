# Security

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub's
[security advisories](https://github.com/visin-platform/visin-py/security/advisories/new) for this
repository. Include what you found, how to reproduce it, and which version you tested.

You will get a reply within a week. A fix is released as a patch version, and the advisory is
published with it.

## Supported versions

Fixes go into the latest release. Upgrading is the fix for older versions.

## What this package does with your credentials

- The token is sent only as an `Authorization: Bearer` header, and only to the configured `VISIN_URL`.
- File uploads go to a signed URL from a separate HTTP session, which never carries the token.
- The token is never written to disk. Reports kept for `visin sync` hold request bodies only, and
  `sync` sends them with the credentials configured when it runs.
- TLS verification is on unless `VISIN_VERIFY_SSL=0` is set.
