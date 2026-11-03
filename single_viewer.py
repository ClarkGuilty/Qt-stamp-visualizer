#!/usr/bin/env python3
"""1-by-1 viewer: classify one object at a time, with every band and survey cutouts.

Run `python single_viewer.py --help` for the options. The code is in qtstamp/single_viewer.py;
installed with pip, the same tool is the `qtstamp-single` command.
"""
from qtstamp.single_viewer import main

if __name__ == "__main__":
    main()
