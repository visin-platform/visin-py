# Errors

A run raises only when it starts (Visin refuses the run, or there is nothing to attach to) and once
it is going, nothing raises into a training loop unless `strict=True`. These are what those raise,
and what `Api` and `sync` raise.

::: visin.errors
    options:
      show_root_heading: false
      members: [VisinError, ConfigurationError, ApiError, TransportError, SyncInProgressError]
