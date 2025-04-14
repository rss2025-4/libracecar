#!/bin/bash -l

source /etc/bash.bashrc

set -eux

cd ../..

colcon build --symlink-install
set +eux
source ./install/setup.bash
set -eux

cd ./src/libracecar

pytest
