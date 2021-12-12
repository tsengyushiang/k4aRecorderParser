import json,cv2,os
import numpy as np
from tqdm import tqdm
import shutil
from glob import glob

maskParentfolder=[
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\1",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\2",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\3",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\4",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\5",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\6",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\7",
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\8"
]

maskfolderkey = "mask"
for parentfolder in maskParentfolder:
    maskfolder = os.path.join(parentfolder,maskfolderkey)
    maskoutput = os.path.join(parentfolder,"mask_mostCenter")
    os.makedirs(maskoutput)
    files = glob(os.path.join(maskfolder,'*.png'))
    for file in tqdm(files):

        mask = cv2.imread(file,0)
        biggestContourMask = np.zeros_like(mask)
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find contour center most close image center
        imageCenter = np.array([mask.shape[1]//2,mask.shape[0]//2])
        dist = []
        for contour in contours:            
            contour = contour.reshape(-1,2)
            contur2center = np.linalg.norm(contour-imageCenter,axis=1)
            dist.append(np.min(contur2center))        
        
        # draw new mask
        index = np.argmin(np.array(dist))        
        biggestContour = contours[index]
        cv2.drawContours(biggestContourMask,[biggestContour],-1,(255,255,255),-1)
        biggestContourMask = cv2.cvtColor(biggestContourMask,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(maskoutput,os.path.basename(file)),biggestContourMask)    
