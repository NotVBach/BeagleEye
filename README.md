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
Logitech C270 ──USB── BeagleBone Black Rev C ──── Ethernet
                               │
                     Power cable (or micro USB for power)
```

## How to run

```bash
./build.sh
```
