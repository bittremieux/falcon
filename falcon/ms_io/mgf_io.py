import math
import os
import re
from typing import Dict, IO, Iterable, List, Union

import numba as nb
import numpy as np
import pyteomics.mgf
import spectrum_utils.spectrum as sus

from ..cluster import similarity
from ..cluster import cluster


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
        base_filename = os.path.splitext(os.path.basename(f_in.name))[0]

        for spectrum_i, spectrum_dict in enumerate(f_in):
            params = spectrum_dict.get("params", {})

            # Prefer original filename from params if available
            filename = os.path.splitext(
                os.path.basename(params.get("filename", base_filename))
            )[0]

            # Build USI-inspired title
            if "scans" in params:
                params["title"] = f"{filename}:scan:{params['scans']}"
            elif "scan" in params:
                params["title"] = f"{filename}:scan:{params['scan']}"
            else:
                title = params.get("title", "")
                if not USI_PATTERN.match(title):
                    params["title"] = f"{filename}:index:{spectrum_i}"

            try:
                yield _parse_spectrum(spectrum_dict)
            except (ValueError, KeyError):
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
    retention_time = float(spectrum_dict["params"].get("rtinseconds", -1))

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
    filename: str, spectra: List[similarity.SpectrumTuple]
) -> None:
    """
    Write the given spectra to an MGF file.

    Parameters
    ----------
    filename : str
        The MGF file name where the spectra will be written.
    spectra : List[similarity.SpectrumTuple]
        The spectra to be written to the MGF file.
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
            "pepmass": spectrum.precursor_mz,
        }
        if not math.isnan(spectrum.precursor_charge):
            params["charge"] = spectrum.precursor_charge
        if hasattr(spectrum, "retention_time"):
            params["rtinseconds"] = spectrum.retention_time
        if hasattr(spectrum, "scan"):
            params["scan"] = spectrum.scan
        if hasattr(spectrum, "cluster_size"):
            params["cluster_size"] = spectrum.cluster_size
        yield {
            "params": params,
            "m/z array": spectrum.mz,
            "intensity array": _scale_intensities(spectrum.intensity),
        }


@nb.njit(cache=True, fastmath=True)
def _scale_intensities(intensity: List[float]) -> List[float]:
    """
    Scale the intensities to the range [0, 1000].

    Parameters
    ----------
    intensity : List[float]
        The intensity array to be scaled.

    Returns
    -------
    List[float]
        The scaled intensities.
    """
    return intensity * 1000 / np.max(intensity)
