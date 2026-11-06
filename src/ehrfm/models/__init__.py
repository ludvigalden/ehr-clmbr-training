"""FEMR/CLMBR model loading, representations and task heads.

Importing this package installs the femr compatibility shims (see
:mod:`ehrfm.models._compat`) before any submodule imports femr's model code.
"""

from ehrfm.models import _compat

_compat.install()
