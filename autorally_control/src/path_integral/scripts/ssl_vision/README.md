# Setting up SSL Vision
This README has some additional documentation to the [ssl-vision wiki](https://github.com/RoboCup-SSL/ssl-vision/wiki)
as well as some steps specific to the setup configured in the VCR lab at UBC.

1. Install and startup (with `flycap` from command line) the [flycap SDK](https://www.flir.ca/products/flycapture-sdk/) to ensure the Flea3 FL3-FW-03S1C camera model works
2. Clone and install the [apriltag repo](https://github.com/AprilRobotics/apriltag)
3. Check to see the *LD_LIBRARY_PATH* env variable includes the path where the **libapriltag.so** files were installed
4. Clone the [ssl-vision repo](https://github.com/RoboCup-SSL/ssl-vision/tree/apriltags) and `git checkout` the [apriltags](https://github.com/RoboCup-SSL/ssl-vision/tree/apriltags) branch
5. Follow the ssl-vision [software requirements steps](https://github.com/RoboCup-SSL/ssl-vision/tree/apriltags#software-requirements)
6. Follow the ssl-vision [compilation steps](https://github.com/RoboCup-SSL/ssl-vision/tree/apriltags#compilation) and use the following command for the cmake step:
`cmake -DUSE_FLYCAP=true -DUSE_APRILTAG=ON ..`
7. Execute `./bin/vision` from the ssl-vision root directory
8. Go to *RoboCup SSL Multi-Cam -> Thread # -> Image Capture -> Capture Control* from there click on *start capture* to start capturing images.
The *Thread #* corresponds to the camera id and will usually be *Thread 0* in a 1-camera setup.
9. Go to *RoboCup SSL Multi-Cam -> Global -> Field Configuration* and edit the parameters *Field Width* and *Field Length* to match the setup in the lab (4x6m in the case of VCR lab)

10. Follow the steps on [camera calibration](https://github.com/RoboCup-SSL/ssl-vision/wiki/camera-calibration#update-control-points) starting from the **Update Control Points** step

11. Go to *RoboCup SSL Multi-Cam -> Thread # -> Visualization* and make sure the following are set to True
    - enable
    - image
    - greyscale (optional, ssl-vision converts image to greyscale before doing line detection)
    - calibration result to make sure the camera calibration result looks reasonable
    - detected AprilTags

12. Go to *RoboCup SSL Multi-Cam -> Thread # -> AprilTag*
    - check the box next to enable
    - select the Tag Family corresponding to the tags being used ([discussion on tag family](https://berndpfrommer.github.io/tagslam_web/making_tags/))
    (also [here](https://github.com/AprilRobotics/apriltag-imgs) is the link where tags can be downloaded and printed )
    - play around with the parameters *decimate* and *blur* until tags are being detected (values 0.5 and 0.8 respectively, seem to work well)
    
13. Go to *RoboCup SSL Multi-Cam -> Global -> Blue/Yellow April Tags* and click on *Add Tag*. Then go to *Unset* and set the value of the tag id.
Repeat this step for all the tags in the camera frame. Note, this step is required for the position data to be part of the UDP packet.

# sensor_noise.py
TODO: add doc

# TrackBots
The repo [rdesc/TrackBots](https://github.com/rdesc/TrackBots) is a forked repo which does the following:
- listen for ssl vision protobuf messages
- use a Kalman filter to get tracking information
- create a ros topic for each detected robot
- for each ros topic, publish the respective robots tracking information which includes pose estimates and velocities