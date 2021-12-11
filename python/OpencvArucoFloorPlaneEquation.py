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
    
    def getAruco2dcoords(self,image):
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)
        cornerDict = {}
        for index,id in enumerate(ids):
            corner = corners[index]
            cornerDict[id[0]]=np.array(corner).astype(int).reshape(-2,2)

        return cornerDict

    def getArucoConvexRegion(self,frameIndex):
        colorPath = os.path.join(self.foldername,self.frames[frameIndex]['color'])

        color = cv2.imread(colorPath)        
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        corners2d_id_dict = self.getAruco2dcoords(color)

        coordArray = np.array([])
        for key in corners2d_id_dict:
            coords = corners2d_id_dict[key]
            for coord in coords:
                coordArray=np.append(coordArray,coord)
        coordArray = coordArray.reshape(-1,2).astype(int)

        arucoRegionMask = np.zeros_like(color)
        for i in range(len(coordArray)):
            for j in range(i+1,len(coordArray)):
                cv2.line(arucoRegionMask, (coordArray[i][0],coordArray[i][1]), (coordArray[j][0],coordArray[j][1]), (255, 255, 255), 1)

        arucoRegionMask = cv2.cvtColor(arucoRegionMask,cv2.COLOR_BGR2GRAY)
        contours,hierarchy = cv2.findContours(arucoRegionMask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arucoRegionMask,contours,-1,(255,255,255),thickness=-1)

        depthPath = os.path.join(self.foldername,self.frames[frameIndex]['depth'])
        depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)*(1e-3)
        depth = np.expand_dims(depth,axis=2) # (1080,1920)->(1080,1920,1)
        pcds = np.concatenate([self.mapping_2d_to_3d_talbe, np.ones_like(depth)],2) #(1080,1920,2)->(1080,1920,3) with coord z       
        pcds = pcds*depth # xy_table to points

        inRegionPoint = np.where((arucoRegionMask>0))
        pointArray = pcds[inRegionPoint]
        pointArray = pointArray[np.where(pointArray[:,-1]>0.5)]

        return pointArray

    def getArucoPoints(self,frameIndex):
        colorPath = os.path.join(self.foldername,self.frames[frameIndex]['color'])
        depthPath = os.path.join(self.foldername,self.frames[frameIndex]['depth'])

        color = cv2.imread(colorPath)        
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)*(1e-3)
        
        depth = np.expand_dims(depth,axis=2) # (1080,1920)->(1080,1920,1)
        pcds = np.concatenate([self.mapping_2d_to_3d_talbe, np.ones_like(depth)],2) #(1080,1920,2)->(1080,1920,3) with coord z       
        pcds = pcds*depth # xy_table to points

        # f = open( os.path.join(self.foldername,f"{self.frames[frameIndex]['depth'].split('.')[0]}.aurco.obj"), 'w')

        corners2d_id_dict = self.getAruco2dcoords(color)
        corners3d_id_dict = {}
        for id in corners2d_id_dict:
            corner2d = corners2d_id_dict[id]            
            corner3d = pcds[corner2d[:,1], corner2d[:,0]]  
            corners3d_id_dict[id] = corner3d

            # for p in corner3d:
            #     line = np.array2string(p, formatter={'float_kind':lambda x: "%.2f" % x})
            #     line = line.replace("[","v ").replace("]","\n")
            #     f.write(line)

            # print(corners3d_id_dict)

        # f.close()

        # points array
        pointArray = np.array([])
        for key in corners3d_id_dict:
            points = corners3d_id_dict[key]
            for point in points:
                pointArray=np.append(pointArray,point)
        pointArray = pointArray.reshape(-1,3)
        pointArray = pointArray[np.where(pointArray[:,-1]>0.5)]

        return corners2d_id_dict, corners3d_id_dict, pointArray

    def fit_plane(self, xyz,z_pos=None):
        """
        if z_pos is not None, the sign
        of the normal is flipped to make 
        the dot product with z_pos (+).
        """
        mean = np.mean(xyz,axis=0)
        xyz_c = xyz - mean[None,:]
        l,v = np.linalg.eig(xyz_c.T.dot(xyz_c))
        abc = v[:,np.argmin(l)]
        d = -np.sum(abc*mean)
        # unit-norm the plane-normal:
        abcd =  np.r_[abc,d]/np.linalg.norm(abc)
        # flip the normal direction:
        if z_pos is not None:
            if np.sum(abcd[:3]*z_pos) < 0.0:
                abcd *= -1
        return abcd

    def getPlaneEquationFromFrame(self,frameIndex,useConvexRegion=True):
        if useConvexRegion:
            pcds = self.getArucoConvexRegion(frameIndex)
        else:
            _, _, pcds = self.getArucoPoints(frameIndex)
        abcd = self.fit_plane(pcds,np.array([0,0,0]))
        return abcd

    def savePointCloudwofloor(self,frameIndex,planeEqABCD,point2planethreshold = 3e-2):
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
        
        validIndex = np.where(pcds[:,-1]>0.5) # get indexing of point depth>0.5
        validPcd = pcds[validIndex]
        validPcdColor = pointcolor[validIndex]

        w = np.ones((validPcd.shape[0],1))
        pcds = np.hstack((validPcd,w)) # (N,4) xyzw: (x,y,z,1.0)
        pcd2planeDistance = abs(pcds@abcd)

        noneFloorPcdIndex = np.where(pcd2planeDistance>point2planethreshold)
        validPcdwoFloor = validPcd[noneFloorPcdIndex]
        validPcdwoFloorColor = validPcdColor[noneFloorPcdIndex]

        pcd_rgb = np.concatenate([validPcdwoFloor, validPcdwoFloorColor],axis=1) # (N,6) xyzrgb
        
        # save to file
        f = open( os.path.join(self.foldername,f"{self.frames[frameIndex]['depth'].split('.')[0]}.wofloor.obj"), 'w')
        for p in tqdm(pcd_rgb):
            line = np.array2string(p, formatter={'float_kind':lambda x: "%.2f" % x})
            line = line.replace("[","v ").replace("]","\n")
            f.write(line)
        f.close()

config = AzurekinectConfig(r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\4\1\config.json")

abcd = config.getPlaneEquationFromFrame(0)
config.savePointCloudwofloor(0,abcd,point2planethreshold=1e-2)