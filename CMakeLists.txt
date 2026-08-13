cmake_minimum_required(VERSION 3.0.2)
PROJECT_NAME = "thread_reconstruction"
project(${PROJECT_NAME})

# Get the information about this package's buildtime dependencies
find_package(ament_cmake REQUIRED COMPONENTS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    message_generation
)

# # Declare the message files to be built
# add_message_files(FILES
# MyMessage1.msg
# MyMessage2.msg
# )
install(
  PROGRAMS
  launch/publish_images.launch.py
  DESTINATION lib/${PROJECT_NAME}
)

# Declare the service files to be built
rosidl_generate_interfaces("thread_reconstruction"
    "Reconstruct.srv"
    "RecordPSMPath.srv"
    "Grasp.srv"
    "TraceThread.srv"
    "msg/BSpline.msg"
    "msg/ThreadSpecs.msg"
    DEPENDENCIES std_msgs geometry_msgs
)

# Actually generate the language-specific message and service files
# generate_messages(DEPENDENCIES std_msgs geometry_msgs)

# Declare that this catkin package's runtime dependencies
# catkin_package(CATKIN_DEPENDS message_runtime)
ament_package()


# include_directories(
# #   include
#   ${catkin_INCLUDE_DIRS}
# )