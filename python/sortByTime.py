import json,cv2,os
import numpy as np
from tqdm import tqdm
import shutil

class Intrinsic:
    def __init__(self,dict):
        self.codx = dict["codx"]
        self.cody = dict["cody"]
        self.cx = dict["cx"]
        self.cy = dict["cy"]
        self.fx = dict["fx"]
        self.fy = dict["fy"]
        self.k1 = dict["k1"]
        self.k2 = dict["k2"]
        self.k3 = dict["k3"]
        self.k4 = dict["k4"]
        self.k5 = dict["k5"]
        self.k6 = dict["k6"]
        self.p1 = dict["p1"]
        self.p2 = dict["p2"]
        self.metric_radius = dict["metric_radius"]

class AzurekinectConfig:
    def __init__(self,filename):
        self.foldername = os.path.dirname(filename)
        self.filename = filename
        with open(filename, 'r',encoding="utf-8") as reader:
            data = json.loads(reader.read())

        # print(data.keys())
        self.width = data["width"]
        self.height = data["height"]

        # filter missing depth or color
        self.frames=[]
        for frame in data["frames"]:
            if "color" in frame and "depth" in frame:
                self.frames.append(frame)
                
        self.intrinsic = Intrinsic(data["intrinsic"])
        self.mapping_2d_to_3d_talbe = np.array(data["mapping_2d_to_3d_table"]).reshape(self.height, self.width,2)        

    def getFrame(self,frameIndex):
        colorPath = os.path.join(self.foldername,self.frames[frameIndex]['color'])
        depthPath = os.path.join(self.foldername,self.frames[frameIndex]['depth'])
        maskPath = os.path.join(self.foldername,'mask_mostCenter',self.frames[frameIndex]['color'])
        return colorPath,depthPath,maskPath

camsMKV=[
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\1\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\2\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\3\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\4\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\5\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\6\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\7\config.json",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\8\config.json"
]

outputfolder = r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\8cam"

configs=[]
framenumber = []
print('loading frames...')
for mkv in tqdm(camsMKV) :
    config = AzurekinectConfig(mkv)
    framenumber.append(len(config.frames))
    configs.append(config)

print('copy frames to time folder.')
frame = np.min(np.array(framenumber))
for t in tqdm(range(frame)):
    # create time folder
    timefolder = os.path.join(outputfolder,str(t))
    colorRawfolder = os.path.join(timefolder,"images")
    os.makedirs(colorRawfolder)
    colorfolder = os.path.join(timefolder,"images_1")
    os.makedirs(colorfolder)
    maskfolder = os.path.join(timefolder,"masks_1")
    os.makedirs(maskfolder)
    for idx,config in enumerate(configs):
        colorPath,_,maskPath = config.getFrame(t)
        
        shutil.copyfile(maskPath, os.path.join(maskfolder,f"{idx}.png"))
        shutil.copyfile(colorPath, os.path.join(colorfolder,f"{idx}.png"))
        shutil.copyfile(colorPath, os.path.join(colorRawfolder,f"{idx}.png"))