import logging
import os
from typing import Dict, IO, Iterable, Union

import pyteomics.mzxml
import spectrum_utils.spectrum as sus
from lxml.etree import LxmlError


logger = logging.getLogger('falcon')


def get_spectra(source: Union[IO, str]) -> Iterable[sus.MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzXML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzXML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterable[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    with pyteomics.mzxml.MzXML(source) as f_in:
        filename = os.path.splitext(os.path.basename(f_in.name))[0]
        try:
            for spectrum_dict in f_in:
                if int(spectrum_dict.get('msLevel', -1)) == 2:
                    # USI-inspired cluster identifier.
                    scan_nr = int(spectrum_dict['id'])
                    spectrum_dict['id'] = f'{filename}:scan:{scan_nr}'
                    try:
                        yield _parse_spectrum(spectrum_dict)
                    except (ValueError, KeyError):
                        pass
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


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
    spectrum_id = spectrum_dict['id']
    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict.get('retentionTime', -1)

    precursor_mz = spectrum_dict['precursorMz'][0]['precursorMz']
    if 'precursorCharge' in spectrum_dict['precursorMz'][0]:
        precursor_charge = spectrum_dict['precursorMz'][0]['precursorCharge']
    else:
        raise ValueError('Unknown precursor charge')

    return sus.MsmsSpectrum(spectrum_id, precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)
