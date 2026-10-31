# Beagle Eye

A simple face detection and recognition program that runs on a BeagleBone Black.

## How it works

```
Camera stream ──> Face detection ──> Face recognition
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
                Known face                               Unknown face
                     │                                         │
                     v                                         v
     Draw bounding box with label             Save automatically to library
```

## Hardware setup

```
Logitech C270 ──USB── BeagleBone Black Rev C ──── Ethernet ──── Host machine
                               │
                     Power cable (or micro USB for power)
```

### How to run
- Run this script on host machine to build locally then the program will be send to device to run as service
```bash
./build.sh
```
- Server can be access using web browser from any device within LAN `http://beaglebone.local:8080/`
- To stop ssh to device, run
```bash
sudo systemctl stop beagleeye
```
