"""How the package works: not part of its public API, and free to change in any release.

config      settings from the environment and arguments
transport   HTTP: retries that know which requests may repeat, errors, signed uploads
serialize   NumPy scalars, tensors and NaN into JSON Visin accepts
reports     each kind of report as data, and how it is delivered
sender      the background thread reports are sent from, in order
spool       reports kept on disk, and sending them later
"""
