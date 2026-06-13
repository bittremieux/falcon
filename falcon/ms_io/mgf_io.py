import math
import os
import re
from typing import Dict, IO, Iterable, List, Union

import numba as nb
import numpy as np
import pyteomics.mgf
import spectrum_utils.spectrum as sus

from ..cluster import cluster
from . import reader_utils

USI_PATTERN = re.compile(r"^mzspec:[^:\s]+:[^:\s]+:(scan:\d+|\d+)(:[^:\s]+)?$")


def get_spectra(source: Union[IO, str]) -> Iterable[sus.MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given MGF file.

    Parameters
    ----------
    source : Union[IO, str]
        The MGF source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    with pyteomics.mgf.MGF(source) as f_in:
        base = reader_utils.base_filename(source, f_in)

        for spectrum_i, spectrum_dict in enumerate(f_in):
            params = spectrum_dict.get("params", {})

            # Prefer original filename from params if available
            filename = os.path.splitext(
                os.path.basename(params.get("filename", base))
            )[0]

            if "title" not in params or not (
                USI_PATTERN.match(params["title"])
                or params["title"].startswith(f"{filename}:cluster:")
            ):
                if "scans" in params:
                    usi = f"{filename}:scan:{params['scans']}"
                elif "scan" in params:
                    usi = f"{filename}:scan:{params['scan']}"
                else:
                    usi = f"{filename}:index:{spectrum_i}"
                params["title"] = usi

            try:
                yield _parse_spectrum(spectrum_dict)
            except (ValueError, KeyError) as e:
                reader_utils.log_skipped_spectrum(
                    source, params.get("title"), e
                )
                continue


def _parse_spectrum(spectrum_dict: Dict) -> sus.MsmsSpectrum:
    """
    Parse the Pyteomics cluster dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics cluster dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed cluster.
    """
    identifier = spectrum_dict["params"]["title"]

    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]
    retention_time = float(
        spectrum_dict["params"].get("rtinseconds", float("nan"))
    )

    precursor_mz = float(spectrum_dict["params"]["pepmass"][0])
    if "charge" in spectrum_dict["params"]:
        precursor_charge = int(spectrum_dict["params"]["charge"][0])
    else:
        precursor_charge = None

    return sus.MsmsSpectrum(
        identifier,
        precursor_mz,
        precursor_charge,
        mz_array,
        intensity_array,
        None,
        retention_time,
    )


def write_spectra(
    filename: str, spectra: List[cluster.ConsensusTuple]
) -> None:
    """
    Write the given spectra to an MGF file.

    Parameters
    ----------
    filename : str
        The MGF file name where the spectra will be written.
    spectra : List[cluster.ConsensusTuple]
        The representative spectra to be written to the MGF file.
    """
    with open(filename, "w") as f_out:
        pyteomics.mgf.write(
            _spectra_to_dicts(spectra),
            f_out,
            use_numpy=True,
        )


def _spectra_to_dicts(
    spectra: List[cluster.ConsensusTuple],
) -> Iterable[Dict]:
    """
    Convert MsmsSpectrum objects to Pyteomics MGF cluster dictionaries.

    Parameters
    ----------
    spectra : List[cluster.ConsensusTuple]
        The spectra to be converted to Pyteomics MGF dictionaries.

    Returns
    -------
    Iterable[Dict]
        The given spectra as Pyteomics MGF dictionaries.
    """
    for i, spectrum in enumerate(spectra):
        params = {
            "title": f"falcon:cluster:{spectrum.cluster_id}",
            "scans": i + 1,
            "pepmass": float(spectrum.precursor_mz),
        }
        if not math.isnan(float(spectrum.precursor_charge)):
            params["charge"] = int(spectrum.precursor_charge)
        if not math.isnan(float(spectrum.retention_time)):
            params["rtinseconds"] = float(spectrum.retention_time)
        yield {
            "params": params,
            "m/z array": spectrum.mz,
            "intensity array": _scale_intensities(spectrum.intensity),
        }


@nb.njit(cache=True, fastmath=True)
def _scale_intensities(intensity: np.ndarray) -> np.ndarray:
    """
    Scale the intensities to the range [0, 1000].

    Parameters
    ----------
    intensity : np.ndarray
        The intensity array to be scaled.

    Returns
    -------
    np.ndarray
        The scaled intensities.
    """
    max_i = np.max(intensity) if intensity.size else np.float32(0.0)
    if max_i <= 0.0:
        return intensity
    return intensity * (np.float32(1000.0) / max_i)
