# EEG_PTZ

Python script analysing EEG signals before and after PTZ injection:
 - Filters the signal using band-stop method and repairs artifacts ([ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html))
 - Generates power spectrum plot using Welch method
 - Computes an averaged Z-Score to generate a threshold and fetches events strictly above it
 - Creates an averaged Hilbert envelope and projects it to other EEG signals to then fetch abnormal events

This pipeline is optimised for Python 3.10.10 and above and was tested on Windows and macOS ARM.

All dependencies are listed in [requirements](requirements.txt).
