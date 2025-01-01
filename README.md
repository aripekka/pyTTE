# pyTTE v.1.0
Package to calculate diffraction curves of deformed crystals by numerically integrating the 1D Takagi-Taupin equation. For derivative work please cite:

Ari-Pekka Honkanen and Simo Huotari, "General method to calculate the elastic deformation and X-ray diffraction properties of bent crystal wafers." IUCrJ 8.1 (2021): 102-115. https://doi.org/10.1107/S2052252520014165

### Requirements
PyTTE works on both Python 3 and Python 2. However, Python 2 support is not guaranteed on future versions. 

For crystallographic data and structure factor calculations PyTTE uses `xraylib` whose installing instructions are found at https://github.com/tschoonj/xraylib. PyTTE is developed and tested on version 4.0.0. In addition, `numpy` (>=1.16.6), `scipy`(>=1.2.1), `multiprocess` (>=0.70.9) and `matplotlib` (>=2.2.3) are required. Contents of testing environments are provided in the repository.

### Installation

PyTTE can be installed using `pip`:

```
pip install pyTTE
```

Note that `xraylib` can not be installed via pip.

Alternatively, the PyTTE works also directly by copying the PyTTE folder to the working directory if the requirements are met.

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

Comprehensive instructions are presented in the docstrings of `TTcrystal`, `TTscan` and `TakagiTaupin` and in the Jupyter notebooks in `examples/`.
