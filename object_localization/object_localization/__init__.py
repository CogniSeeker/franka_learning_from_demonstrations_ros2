
import os
path = os.path.dirname(os.path.abspath(__file__))
# OBJECT_LOCALIZATION_PATH points to the package source dir (holds cfg/ and
# config/); without it the ROS2 build dir is used (see franka_hri README)
package_path = os.path.expanduser(os.environ.get("OBJECT_LOCALIZATION_PATH", "/".join(path.split("/")[:-1])))