#!/usr/bin/env python3
"""Lobby: set up a session and start the mosaic, the 1-by-1 viewer, or both in a chain.

Run `python lobby.py --help` for the options. The code is in qtstamp/lobby.py;
installed with pip, the same tool is the `qtstamp-lobby` command.
"""
from qtstamp.lobby import main

if __name__ == "__main__":
    main()
