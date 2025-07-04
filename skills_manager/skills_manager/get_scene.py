#!/usr/bin/env python3
"""
Playback of trajectories and storing them into a databaseself.

see scene at /scene topic: 
```
ros2 topic echo /scene
```
"""
from skills_manager.lfd import LfD
from scene_getter.scene_getting import SceneGetter
from pointing_object_selection.pointing_experiment.utils import print_table_scene
import rclpy, time

class ExtLfD(SceneGetter, LfD):
    def __init__(self):
        super(ExtLfD, self).__init__()

def main():
    rclpy.init()
    lfd = ExtLfD()
    lfd.start()
    lfd.start_publishing_scene()

    while rclpy.ok():
        time.sleep(1.0)
        print_scene = lfd.scene.copy()
        print_table_scene(print_scene, [], [0.,0.,0.])
        

if __name__ == '__main__':
    main()