#!/usr/bin/env bash
set -Eeuo pipefail

# ROS setup scripts probe optional variables and are not nounset-safe.
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO:?ROS_DISTRO must be set}/setup.bash"
set -u

exec "$@"
