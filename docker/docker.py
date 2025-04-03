#!/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

compose_cmd = [
    "docker",
    "compose",
    *["-f", "docker-compose-sim-alan.yml"],
]

if len(sys.argv) > 1:
    os.execlp("docker", *compose_cmd, *sys.argv[1:])
    assert False

# drvs = [
#     "n#cmake-format",
#     "n#html-tidy",
#     "n#ripgrep",
#     "n#shfmt",
#     "n#taplo",
#     "p#p.emacs30-gtk3",
#     "p#nixwrapper",
#     # "p#pdf-tools-epdfinfo",
#     "p#prettierd",
#     "p#pythontools",
# ]


drvs = [
    "n#cmake-format",
    "n#direnv",
    "n#html-tidy",
    "n#ripgrep",
    "n#shfmt",
    "n#taplo",
    "n#emacs30-gtk3",
    "p#less-build.less-download-flakes.nixwrapper",
    # "p#less-build.less-download-flakes.pdf-tools-epdfinfo",
    "p#pythontools",
    "p#js",
    # "p#poetry",
]


res = (
    subprocess.run(
        ["nix", "build", *drvs, "--no-link", "--print-out-paths"],
        check=True,
        stdout=subprocess.PIPE,
    )
    .stdout.decode()
    .removesuffix("\n")
)
# print(res)
extra_path = ":".join([f"{x}/bin" for x in res.split("\n")])
print("extra_path:", extra_path)


subprocess.run(
    [*compose_cmd, "up", "--build", "-d"],
    check=True,
)

os.execlp(
    "docker",
    *compose_cmd,
    "exec",
    "-it",
    "racecar",
    "bash",
    "-c",
    f'exec env PATH="{extra_path}:${{PATH}}" bash -l',
)
