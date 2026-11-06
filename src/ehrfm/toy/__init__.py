"""Seeded synthetic inputs and a toy CLMBR checkpoint for hermetic runs.

:mod:`ehrfm.toy.artifacts` writes a small MEDS extract and label set that
follow the input contract; :mod:`ehrfm.toy.checkpoint` writes a tiny
checkpoint directory in the released CLMBR layout. Both are generated into a
caller-chosen directory at run time; the test suite and ``ehrfm pretrain
--toy`` use them, and nothing they produce is committed.
"""
