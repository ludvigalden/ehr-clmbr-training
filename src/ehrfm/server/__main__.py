"""``python -m ehrfm.server``: serve the configured models.

Configuration comes from the ``EHRFM_SERVER_`` prefixed environment
variables (see :class:`ehrfm.server.config.ServerConfig`); pass
``--help`` to see the same options as flags via ``ehrfm serve``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

from ehrfm.server.app import create_app
from ehrfm.server.config import ServerConfig


def main(argv: list[str] | None = None) -> int:
    """Parse the minimal ``--toy``/``--work-dir`` flags and run uvicorn."""
    toy = False
    work_dir: Path | None = None
    rest: list[str] = []
    args = sys.argv[1:] if argv is None else argv
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--toy":
            toy = True
        elif argument == "--work-dir":
            index += 1
            if index >= len(args):
                print("error: --work-dir needs a value", file=sys.stderr)
                return 2
            work_dir = Path(args[index])
        elif argument.startswith("--work-dir="):
            work_dir = Path(argument.split("=", 1)[1])
        else:
            rest.append(argument)
        index += 1
    if rest:
        print(f"error: unexpected arguments: {' '.join(rest)}", file=sys.stderr)
        return 2
    config = ServerConfig.from_env()
    if toy:
        config = config.model_copy(update={"toy": True, "toy_work_dir": work_dir})
    elif work_dir is not None:
        print("error: --work-dir needs --toy", file=sys.stderr)
        return 2
    uvicorn.run(create_app(config), host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
