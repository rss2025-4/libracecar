#!/bin/env python

import os
import subprocess
from pathlib import Path

os.chdir(Path(__file__).parent)

subprocess.run(
    "sphinx-apidoc --separate -o source ../libracecar/", shell=True, check=True
)

# subprocess.run("make clean", shell=True, check=True)
subprocess.run("make html", shell=True, check=True)
