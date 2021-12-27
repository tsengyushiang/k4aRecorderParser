# Catalog

- [Convert MKV to RGB and aligned Depth](./c++/k4aMKVparser)
- [Detectron2 human segmentation](./python/detectron2humanseg.py)

# Record

- [Offical document](https://docs.microsoft.com/zh-tw/azure/kinect-dk/azure-kinect-recorder)
```
k4arecorder [options] output.mkv

 Options:
  -h, --help              Prints this help
  --list                  List the currently connected K4A devices
  --device                Specify the device index to use (default: 0)
  -l, --record-length     Limit the recording to N seconds (default: infinite)
  -c, --color-mode        Set the color sensor mode (default: 1080p), Available options:
                            3072p, 2160p, 1536p, 1440p, 1080p, 720p, 720p_NV12, 720p_YUY2, OFF
  -d, --depth-mode        Set the depth sensor mode (default: NFOV_UNBINNED), Available options:
                            NFOV_2X2BINNED, NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, PASSIVE_IR, OFF
  --depth-delay           Set the time offset between color and depth frames in microseconds (default: 0)
                            A negative value means depth frames will arrive before color frames.
                            The delay must be less than 1 frame period.
  -r, --rate              Set the camera frame rate in Frames per Second
                            Default is the maximum rate supported by the camera modes.
                            Available options: 30, 15, 5
  --imu                   Set the IMU recording mode (ON, OFF, default: ON)
  --external-sync         Set the external sync mode (Master, Subordinate, Standalone default: Standalone)
  --sync-delay            Set the external sync delay off the master camera in microseconds (default: 0)
                            This setting is only valid if the camera is in Subordinate mode.
  -e, --exposure-control  Set manual exposure value (-11 to 1) for the RGB camera (default: auto exposure)
```
### Sub and Master with external sync cable
```
k4arecorder.exe --external-sync sub -e -8 -r 5 -l 10 sub1.mkv

Device serial number: 000011590212
Device version: Rel; C: 1.5.78; D: 1.5.60[6109.6109]; A: 1.5.13
Device started
[subordinate mode] Waiting for signal from master

k4arecorder.exe --external-sync master -e -8 -r 5 -l 10 master.mkv
```
# FFmpeg parse to normal mp4

```
ffmpeg.exe -i test.mkv -map 0:0 -vsync 0 -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4
```
# Parse Images Quick start

```
PS D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release> .\k4aMKVparser.exe
Params is Required : k4aMKVparser.exe <input.mkv> <folder>
params example :
<input.mkv> : "D:/nerf-data/20121202/8cam/6/device0.mkv"
<folder> : ./output
PS D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release> .\k4aMKVparser.exe "D:\nerf-data\20121202\8cam\6\device3.mkv" ./output/4
```

### Outputs

```
{
    // frame path
    "frames": [
        {
            "depth": "00000001.depth.png"
        },
        ...,
        {
            "color": "00000452.color.png",
            "depth": "00000452.depth.png"
        }
    ],
    "height": 1080,
    "intrinsic": {
        "codx": 0.0,
        "cody": 0.0,
        "cx": 959.7430419921875,
        "cy": 549.100830078125,
        "fx": 911.23876953125,
        "fy": 911.1605834960938,
        "k1": -0.2370704859495163,
        "k2": -2.3210363388061523,
        "k3": 1.7586480379104614,
        "k4": -0.3527301251888275,
        "k5": -2.1014151573181152,
        "k6": 1.645255208015442,
        "metric_radius": 5.324934164434305e-44,
        "p1": -0.0001248455955646932,
        "p2": -3.619360359152779e-05
    },
    "mapping_2d_to_3d_table": [...] // w*h*2 : (x,y)
    "width": 1920
}
```

# Project setup (already done)
### win10 vs2019 Azure Kinect SDK setup

- [Download window SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md)

    - The installer will put all the needed headers, binaries, and tools in the location you choose (by default this is `C:\Program Files\Azure Kinect SDK version\sdk`

    - In my case I copy all files to `C:\git\Azure Kinect SDK version\sdk`

- Open vs2019 

    - Include `C:\git\Azure Kinect SDK v1.4.1\sdk\include`
    
    - Add lib `C:\git\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\lib\k4a.lib`
    - Add lib `C:\git\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\lib\k4arecord.lib`

- Add DLL to System Path

    - `C:\git\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin`

- Tutorials :

    - [vs2019 Setting](https://blog.csdn.net/hanshuning/article/details/112555140)

    - [offical startup](https://pterneas.com/2020/03/19/azure-kinect/)