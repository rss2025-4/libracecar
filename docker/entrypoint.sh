#!/bin/bash

set -eux

/bin/bash -c 'mkdir -p $HOME/.rviz2; cp /default.rviz $HOME/.rviz2'

git config --global user.name "Xinyang Chen"
git config --global user.email "chenxinyang99@gmail.com"

set +eux
source /etc/bash.bashrc
set -eux

# echo "entrypoint.sh: running colcon build --symlink-install"
# /bin/bash -c 'cd $HOME/racecar_ws; colcon build --symlink-install' || true

printf "ready!"

# Hang
tail -f
