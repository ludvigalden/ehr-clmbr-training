"""Typed pins of the CLMBR-T-Base dictionary release.

The values are facts copied from the producing repository's
``src/omopmap/clmbr/pins.toml``; no code from that repository is imported.
Every dictionary acquisition (the pinned Hugging Face revision or a local
copy of the same file) must check ``file_sha256``, and a derived concept set
must check ``concept_list_sha256``. The model is gated: the Hugging Face
route needs a token accepted for the repo's access terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class DictionaryPin:
    """One pinned dictionary release: where it comes from and how it is checked."""

    repo: str
    revision: str
    filename: str
    file_sha256: str
    concept_list_sha256: str
    licence: str


CLMBR_T_BASE_PIN: Final = DictionaryPin(
    repo="StanfordShahLab/clmbr-t-base",
    revision="a0f2b36ef3976faf032bfbce23030cb640b5e97f",
    filename="dictionary.msgpack",
    file_sha256="481ddac70a37e79bbfadde1411ad674544a161e3c1948149989cd159a34203ee",
    concept_list_sha256="71f396167f40096c9ff11dc1b762746e946a7ddaec195f11a813a341bffedbc1",
    licence=(
        "Gated release under the EHRSHOT Credentialed Health Data License; "
        "model card: CC-BY-NC-4.0; bundled LICENSE file: CC BY-NC-SA 4.0"
    ),
)

__all__ = ["CLMBR_T_BASE_PIN", "DictionaryPin"]
