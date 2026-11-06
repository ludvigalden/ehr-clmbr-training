"""Continued pretraining of a CLMBR checkpoint on a MEDS extract.

Importing this package installs the femr compatibility shims through
:mod:`ehrfm.models` before femr's model code is imported.
"""

import ehrfm.models as _models

del _models
