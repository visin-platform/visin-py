# Tokens

Visin has two kinds of key. Both look like `vsn_live_…`, and both go in `VISIN_TOKEN`. Which one you
need depends on what the script does.

| | Pipeline key | Account API key |
| --- | --- | --- |
| Made under | the project's **Settings → Pipeline keys** | **Account → API keys** |
| Limited to | one project | whatever you give it, across projects |
| Writes runs | into its own project, always | into the project you name with `project=` |
| Reads runs | its own project | every project the key's scopes allow |
| Use it for | training and test scripts, jobs, CI runners | notebooks and analysis (`Api`), anything that spans projects |

## For a training job, use a pipeline key

It can write only to its own project, so a leaked key cannot touch anything else, and the job needs
no `project=`: whatever the script names, the server puts the run in the key's project.

```sh
export VISIN_TOKEN=vsn_live_…      # from Settings → Pipeline keys
```

## For analysis, use an account key

`Api` reads across projects with a key that has read scopes. A key that is not limited to a project
needs to be told where to write:

```sh
export VISIN_PROJECT=road-seg      # id or slug
```

## What each command and call needs

| | Needs |
| --- | --- |
| `visin.init`, `Run.attach`, `visin sync` | a key that may write to the project |
| `visin check` | any key; `--write` also needs a project |
| `visin runs`, `Api` | a key that may read the project. Public projects need none |
| `visin datasets`, `visin download` | none for public datasets; a key for private ones |

## Keeping a key safe

- `visin login` saves it in a file only you can read. See [Configuration](configuration.md#where-settings-come-from).
- Do not put it in a script, a config checked into git, or an image. Use the environment, a
  `VISIN_ENV_FILE`, or your scheduler's secrets.
- `visin check` prints only the first and last four characters.
- To revoke a key, revoke or delete it in Visin. Every job using it then fails at `init` with a refusal.

## Something wrong?

`visin check` says which kind of key it was given and whether Visin accepts it. See
[Troubleshooting](troubleshooting.md#starting-a-run-fails).
