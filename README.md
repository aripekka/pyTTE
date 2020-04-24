# pyTTE
Package to calculate diffraction curves of deformed crystals by numerically integrating the Takagi-Taupin equations

### Requirements
In addition to `numpy`, `scipy`, and `matplotlib`, PyTTE requires `xraylib` https://github.com/tschoonj/xraylib

### Installation

PyTTE can be installed using `pip` (NB: v 1.0 is not yet in PyPI):

```
pip install pyTTE
```

Alternatively, the PyTTE works also directly by copying the PyTTE folder to the working directory.

### Example of use

Following snippet calculates the reflectivity of unstrained GaAs(400) reflection in angle domain for sigma-polarized beam at 6 keV. The thickness of the crystal is set to 300 microns and the asymmetry angle to 0. The Debye-Waller factor is 1 (= 0 K). Scan vector is given in units of arc sec (microradians are also supported). Reflectivity and transmission (the latter is not implemented yet, though) are given as the output of .run() and can be quickly plotted with .plot().

```
import numpy as np
import matplotlib.pyplot as plt
from pyTTE import TakagiTaupin, TTcrystal, TTscan, Quantity

ttx = TTcrystal(crystal = 'GaAs', hkl = [4,0,0], thickness = Quantity(300,'um'), debye_waller = 1)
tts = TTscan(constant=Quantity(6,'keV'), scan=Quantity(np.linspace(-10,30,150), 'arcsec'), polarization='sigma')

tt=TakagiTaupin(ttx,tts)

scan_vector, R, T = tt.run()
tt.plot()

plt.show()
```

The code differentiates between the Bragg and Laue cases on the basis of the exit angle of the diffracted beam which, on the other hand, is dictated by the Bragg angle and the asymmetry. To change the previous example to Laue geometry, give `asymmetry = Quantity(90,'deg')` when initializing `TTcrystal`.

Comprehensive instructions are presented in the docstrings of TTcrystal, TTscan and TakagiTaupin and in the Jupyter notebooks in `examples/`.
