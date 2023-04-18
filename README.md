# Analysis pipeline EEG with PTZ injection

Python script analysing EEG signals before and after PTZ injection:
 - Imports .edf files
 - Filters the signal using a notch filter and band-stop method
 - Creates a baseline and after PTZ injection window for each recording
 - Normalizes each recordings using Z-score
 - Compares each baseline Z-score to its corresponding PTZ Z-score
 - Returns the event count during PTZ window that are above baseline and the latency of the first event in minutes

This pipeline is optimised for Python 3.10.11 and above and was tested on both Windows and macOS ARM.

*All dependencies are listed in [requirements](requirements.txt).

# Development

This pipeline was created and is maintained by SJG. Contributions are more than welcome so feel free to open an issue or submit a pull request!

To see the code or report a bug, please visit the [GitHub repository](https://github.com/sjg2203/EEG_PTZ).

Note that this program is provided with NO WARRANTY OF ANY KIND under Apache 2.0 [license](LICENSE).

# Citation

To cite this pipeline, please use this paper as reference:

 - Scekic-Zahirovic J., Benetton C., Brunet A. *et al*., "Noradrenaline deficiency as a driver of cortical hyperexcitability in amyotrophic lateral sclerosis" Science Translational Medicine (2023). doi: https://doi.org/10.7554/eLife.70092
