FROM ubuntu:22.04
ARG ROS_DISTRO=jazzy
ARG GZ_VERSION=harmonic
 
ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release
 
# Install ROS2
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y ros-jazzy-desktop
 
# Install Gazebo Harmonic
RUN apt-get update && apt-get install -y \
    software-properties-common
RUN sh -c '. /etc/os-release && echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $UBUNTU_CODENAME main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN curl http://packages.osrfoundation.org/gazebo.key | apt-key add -


RUN apt-get update  && \
    apt-get update && \
    apt-get install -y gz-$GZ_VERSION ros-$ROS_DISTRO-ros-gz && \
    rm -rf /var/lib/apt/lists/* 
 
# Setup environment
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
 
CMD ["/bin/bash"]