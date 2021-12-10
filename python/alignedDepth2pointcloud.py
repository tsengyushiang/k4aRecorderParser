import json,cv2,os
import numpy as np
from tqdm import tqdm

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
        self.frames = data["frames"]
        self.intrinsic = Intrinsic(data["intrinsic"])
        self.mapping_2d_to_3d_talbe = np.array(data["mapping_2d_to_3d_table"]).reshape(self.height, self.width,2)

    def savePointCloud(self,frameIndex):
        colorPath = os.path.join(self.foldername,self.frames[frameIndex]['color'])
        depthPath = os.path.join(self.foldername,self.frames[frameIndex]['depth'])

        color = cv2.imread(colorPath)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)*(1e-3)
        
        depth = np.expand_dims(depth,axis=2) # (1080,1920)->(1080,1920,1)
        pcds = np.concatenate([self.mapping_2d_to_3d_talbe, np.ones_like(depth)],2) #(1080,1920,2)->(1080,1920,3) with coord z       
        pcds = pcds*depth # xy_table to points
        pcds = pcds.reshape(-1,3) # (N,3) xyz
        pointcolor = color.reshape(-1,3) # (N,3) bgr
        validPcd = np.where(pcds[:,-1]>0.5) # get indexing of point depth>0.5
        pcd_rgb = np.concatenate([pcds[validPcd], pointcolor[validPcd]],axis=1) # (N,6) xyzrgb
        
        # save to file
        f = open( os.path.join(self.foldername,f"{self.frames[frameIndex]['depth'].split('.')[0]}.obj"), 'w')
        for p in tqdm(pcd_rgb):
            line = np.array2string(p, formatter={'float_kind':lambda x: "%.2f" % x})
            line = line.replace("[","v ").replace("]","\n")
            f.write(line)
        f.close()

config = AzurekinectConfig(r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\1\config.json")
config.savePointCloud(100)