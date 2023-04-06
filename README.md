# Analysis pipeline EEG with PTZ injection

Python script analysing EEG signals before and after PTZ injection:
 - Import .edf files
 - Filters the signal using a notch filter and band-stop method
 - Creates a baseline and after PTZ injection window for each recording
 - Computes an averaged for each baseline window
 - Compares each baseline average to its corresponding PTZ window
 - Returns the event count during PTZ window that are above baseline and the latency of the first event

This pipeline is optimised for Python 3.10.10 and above and was tested on Windows and macOS ARM.

All dependencies are listed in [requirements](requirements.txt).

# Citation

To cite this pipeline, please use this paper as reference:

 - Scekic-Zahirovic J., Benetton C., Brunet A. *et al*., "Noradrenaline deficiency as a driver of cortical hyperexcitability in amyotrophic lateral sclerosis" Science Translational Medicine (2023). doi: https://doi.org/10.7554/eLife.70092
