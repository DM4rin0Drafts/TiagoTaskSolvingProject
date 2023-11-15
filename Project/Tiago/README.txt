Tutorial: How to create urdf-files from urdf.xacro

1) Install ros-melodic
2) create a ros workspace
	- source /opt/ros/melodic/setup.bash
	- mkdir -p ~/catkin_ws/src
	- cd ~/catkin_ws/
	- catkin_make
	- source /devel/setup.bash
3) copy Tiago folder from DeepHeuristic Project:
	- Folder has to be copied in src-folder in the catkin_ws/src-folder
4) in catkin/src/-folder:
	- cd to tiago
5) create urdf file:
	- rosrun xacro xacro tiago_description/robots/tiago.urdf.xacro > tiago_single.urdf


6) Important: Everytime you use a new terminal, you HAVE to do the following command in the catkin_ws folder:
	- source /devel/setup.bash


Alternativly, programm it with rospy:
	- it looks like that we can do that directly but it tooks longer for the simulation to load the enviroment
