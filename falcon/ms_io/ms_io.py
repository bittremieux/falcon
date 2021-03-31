import os
from typing import Iterable, Iterator

from spectrum_utils.spectrum import MsmsSpectrum

from . import mzxml_io
from . import mgf_io
from . import mzml_io


def get_spectra(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given file.

    Supported file formats are MGF, MSP, mzML, mzXML.

    Parameters
    ----------
    filename : str
        The file name from which to read the spectra.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    if not os.path.isfile(filename):
        raise ValueError(f'Non-existing peak file {filename}')

    _, ext = os.path.splitext(filename.lower())
    if ext == '.mgf':
        spectrum_io = mgf_io
    elif ext == '.mzml':
        spectrum_io = mzml_io
    elif ext == '.mzxml':
        spectrum_io = mzxml_io
    else:
        raise ValueError(f'Unknown spectrum file type with extension "{ext}"')

    for spec in spectrum_io.get_spectra(filename):
        spec.is_processed = False
        yield spec


def write_spectra(filename: str, spectra: Iterable[MsmsSpectrum]) -> None:
    """
    Write the given spectra to a peak file.

    Supported formats: MGF.

    Parameters
    ----------
    filename : str
        The file name where the spectra will be written.
    spectra : Iterable[MsmsSpectrum]
        The spectra to be written to the peak file.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext == '.mgf':
        spectrum_io = mgf_io
    else:
        raise ValueError('Unsupported peak file format (supported formats: '
                         'MGF)')

    spectrum_io.write_spectra(filename, spectra)
