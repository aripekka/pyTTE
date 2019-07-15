# pyTTE
Package to calculate diffraction curves of deformed crystals by numerically integrating the Takagi-Taupin equations

### Requirements
In addition to `numpy`, `scipy`, and `matplotlib`, PyTTE requires `xraylib` https://github.com/tschoonj/xraylib

### Installation
PyTTE can be installed using `pip`:
```
pip install pyTTE
```
The installation can be tested by running example calculations:
```
import pyTTE.examples
pyTTE.examples.run()
```
The script runs four different test cases (Laue-Bragg, symmetric-asymmetric, unbent-bent...) and should take a minute or so to complete.

### Usage

Following snippet calculates the reflectivity of unstrained GaAs(400) reflection in angle domain for sigma-polarized beam at 6 keV. The thickness of the crystal is set to 300 microns and the asymmetry angle to 0. The Debye-Waller factor is 1 (= 0 K) and the minimum integration step is 1e-10. Scan vector `th` is given in units of arc secs. Reflectivity and transmission (the latter is not implemented yet, though) are given as an output.

```
import numpy as np
from pyTTE import takagitaupin

th = np.linspace(-10,30,150)
R,T=takagitaupin('angle',th,6,'sigma','GaAs',[4,0,0],0,300,None,1,1e-10)
```

For energy scan, `th` is replaced with energy scan vector which is given in meV. The constant energy (6 in the example), is replaced with the (kinematic) Bragg angle (in degrees).

The code differentiates between the Bragg and Laue cases on the basis of the exit angle of the diffracted beam which, on the other hand, is dictated by the Bragg angle and the asymmetry. To change the previous example to Laue geometry, the asymmetry angle is changed from 0 to 90.

Other examples of use can be found in `pyTTE/examples/examples.py`.
