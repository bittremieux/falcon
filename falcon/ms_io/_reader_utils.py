"""Shared helpers for the per-format spectrum readers (mgf/mzml/mzxml).

These intentionally live in a leaf module that imports none of the readers, so
the readers can import it without creating a cycle with ``ms_io``.
"""

import logging
import os
from typing import IO, Union

logger = logging.getLogger("falcon")


def base_filename(source: Union[IO, str], reader) -> str:
    """
    Derive the base file name (no directory, no extension) used to build
    USI-style spectrum identifiers.

    Works for both path strings and open file objects; the latter fall back to
    the ``name`` attribute of the source or of the pyteomics ``reader``.

    Parameters
    ----------
    source : Union[IO, str]
        The source passed to ``get_spectra`` (a path or an open file object).
    reader
        The open pyteomics reader, used as a secondary source of a ``name``.

    Returns
    -------
    str
        The base file name, or ``"unknown"`` when no name can be determined.
    """
    if isinstance(source, str):
        name = source
    else:
        name = getattr(source, "name", None) or getattr(reader, "name", None)
    if not name:
        return "unknown"
    return os.path.splitext(os.path.basename(name))[0]


def log_skipped_spectrum(
    source: Union[IO, str], identifier, exc: Exception
) -> None:
    """
    Record that a single spectrum could not be parsed and is being skipped.

    Logged at ``debug`` rather than ``warning``: a file may contain millions of
    spectra and a per-spectrum warning would flood the log. A failure to read a
    whole file remains a ``warning`` in the individual readers.
    """
    logger.debug(
        "Skipping unparsable spectrum %s in %s: %s", identifier, source, exc
    )
