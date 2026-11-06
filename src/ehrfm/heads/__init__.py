"""Task heads trained on encoder representations inside the experiment pipeline.

Importing :mod:`ehrfm.heads.hooks` registers the ``linear_probe`` head hook
(the pipeline imports it before every run). Trained heads are found in a work
directory with :func:`ehrfm.heads.store.trained_heads` and published for
serving with :class:`ehrfm.inference.registry.HeadRegistry`.
"""
