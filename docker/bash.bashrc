echo "/etc/bash.bashrc : sourcing ROS"
source /opt/ros/$ROS_DISTRO/setup.bash
if [ -e "$HOME/racecar_ws/install/setup.bash" ]; then
	source $HOME/racecar_ws/install/setup.bash
fi
