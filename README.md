# k4aRecorderParser
## Quick start

```
PS D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release> .\k4aMKVparser.exe
Params is Required : k4aMKVparser.exe <input.mkv> <folder>
params example :
<input.mkv> : "D:/nerf-data/20121202/8cam/6/device0.mkv"
<folder> : ./output
PS D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release> .\k4aMKVparser.exe "D:\nerf-data\20121202\8cam\6\device3.mkv" ./output/4
```

## Outputs

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

## win10 vs2019 Azure Kinect SDK setup

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