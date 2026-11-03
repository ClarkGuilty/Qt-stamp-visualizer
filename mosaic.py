#!/usr/bin/env python3
"""Mosaic: classify stamps a page-sized grid at a time, for fast first-pass triage.

Run `python mosaic.py --help` for the options. The code is in qtstamp/mosaic.py;
installed with pip, the same tool is the `qtstamp-mosaic` command.
"""
from qtstamp.mosaic import main

if __name__ == "__main__":
    main()
