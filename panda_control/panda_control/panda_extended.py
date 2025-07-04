from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from scene_getter.scene_lib.scene_object import SceneObject

def top_of_object(obj: SceneObject, safety_z_const: float = 0.3):
    # TODO: Check if object exist
    # TODO: const. yaw rotation of an object? 
    return PoseStamped(header=Header(),
            pose=Pose(position=Point(x=obj.position[0], 
                               y=obj.position[1], 
                               z=obj.position[2] + safety_z_const), 
                orientation=Quaternion(x=1.0, #obj.orientation[0],
                                       y=0.0, #obj.orientation[1],
                                       z=0.0, #obj.orientation[2],
                                       w=0.0) #obj.orientation[3])
                )
            )

