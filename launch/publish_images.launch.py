from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import sys

BASE_DIR="/media/emmah/PortableSSD/Arclab_data"
PARENT_FOLDER="thread_meat_3_21"
TRIAL_NAME="trial_33_video"
IMAGE_DIR=f"{BASE_DIR}/{PARENT_FOLDER}/{TRIAL_NAME}/"
START_FRAME="289"
# END_FRAME="428"
END_FRAME="337"
# START_FRAME="315"
# END_FRAME="315"

HZ="30"
def generate_launch_description():
    trial_path = DeclareLaunchArgument(
        'trial_path', default_value=IMAGE_DIR
    )
    trial_name = DeclareLaunchArgument(
        'trial_name', default_value=TRIAL_NAME
    )
    frame = DeclareLaunchArgument(
        'frame', default_value=START_FRAME
    )
    print('python version in publish script')
    print(sys.version)

    return LaunchDescription([
        trial_path,
        trial_name,
        frame,

        Node(
            package='thread_reconstruction',
            executable='image_publish_node',  # name from setup.py / entry point
            name='image_publish_node',
            output='screen',
            arguments=[
                '--trial_path', IMAGE_DIR,
                '--trial_name', TRIAL_NAME,
                '--start_frame', START_FRAME,
                '--end_frame', END_FRAME,
                '--hz', HZ
                # '--trial_path', LaunchConfiguration('trial_path'),
                # '--trial_name', LaunchConfiguration('trial_name'),
                # '--frame', LaunchConfiguration('frame')
            ]
        )
    ])
 