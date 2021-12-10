import json,cv2,os
import numpy as np
from tqdm import tqdm

'''
ModuleNotFoundError: No module named 'cv2.aruco'

-> pip install opencv-contrib-python
'''

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

    def getArucoPoints(self,frameIndex):
        colorPath = os.path.join(self.foldername,self.frames[frameIndex]['color'])
        depthPath = os.path.join(self.foldername,self.frames[frameIndex]['depth'])

        color = cv2.imread(colorPath)        
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)*(1e-3)
        
        depth = np.expand_dims(depth,axis=2) # (1080,1920)->(1080,1920,1)
        pcds = np.concatenate([self.mapping_2d_to_3d_talbe, np.ones_like(depth)],2) #(1080,1920,2)->(1080,1920,3) with coord z       
        pcds = pcds*depth # xy_table to points

        f = open( os.path.join(self.foldername,f"{self.frames[frameIndex]['depth'].split('.')[0]}.aurco.obj"), 'w')

        corners2d_id_dict = self.getAruco2dcoords(color)
        corners3d_id_dict = {}
        for id in corners2d_id_dict:
            corner2d = corners2d_id_dict[id]            
            corner3d = pcds[corner2d[:,1], corner2d[:,0]]  
            corners3d_id_dict[id] = corner3d

            for p in corner3d:
                line = np.array2string(p, formatter={'float_kind':lambda x: "%.2f" % x})
                line = line.replace("[","v ").replace("]","\n")
                f.write(line)

            print(corners3d_id_dict)

        f.close()

        return corners2d_id_dict, corners3d_id_dict

    def getAruco2dcoords(self,image):
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)
        cornerDict = {}
        for index,id in enumerate(ids):
            corner = corners[index]
            cornerDict[id[0]]=np.array(corner).astype(int).reshape(-2,2)

        return cornerDict
   

config = AzurekinectConfig(r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\4\1\config.json")
config.getArucoPoints(0)