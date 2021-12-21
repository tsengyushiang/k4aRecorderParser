import json,cv2,os
import numpy as np
from tqdm import tqdm
import shutil
from glob import glob

from sklearn.cluster import KMeans,DBSCAN #pip install scikit-learn

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

maskParentfolder=[
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\6"
]

maskfolderkey = "mask_mostCenter"

def depth2Clustering(cropDepth):
    '''
        input : depth map
        output : Mask
        process : kmean to 2 class clustering        
    '''
    depth_in_bbox = cropDepth.copy()

    # ignore invalid depth
    depth_in_bbox_1d = depth_in_bbox.flatten()
    depth_in_bbox_1d_woNotValid = depth_in_bbox_1d[depth_in_bbox_1d!=0]

    n_clusters=2 # class number

    # train without invalid depth
    kmeans = KMeans(n_clusters=n_clusters).fit(depth_in_bbox_1d_woNotValid.reshape(-1,1))
    clustering = kmeans.predict(depth_in_bbox_1d.reshape(-1,1))
    
    nearClusterIndex = np.argmin(kmeans.cluster_centers_)
    # print(kmeans.cluster_centers_,index)

    mask = np.zeros_like(clustering)
    mask[clustering==nearClusterIndex]=255

    mask = mask.reshape(depth_in_bbox.shape[0],depth_in_bbox.shape[1])
    return mask

def outlinerRemoveDBSCAN(colorImg, bkgdMask):
    
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2HSV) # covert to HSV
    img_float = np.float32(colorImg)
    colorImg_flat = img_float.reshape((-1,3))
    bkgdMask_flat = bkgdMask.flatten()
    img_bkgd_flat = colorImg_flat[bkgdMask_flat==0]
   
    dbscan = DBSCAN(eps=3, min_samples=10).fit(img_bkgd_flat)
    
    # found most number index
    idx,count = np.unique(dbscan.labels_,return_counts=True)
    maxCountClusterId = idx[np.argmax(count)]

    # refined forground mask
    mask = bkgdMask_flat.copy()
    maskBG = mask[bkgdMask_flat==0]
    maskBG[dbscan.labels_==-1] = 255
    mask[bkgdMask_flat==0]=maskBG
    mask = mask.astype(np.uint8).reshape((bkgdMask.shape))

    # calc centroid and largest error as threshold later
    points_of_cluster = img_bkgd_flat[dbscan.labels_==maxCountClusterId,:]
    centroid_of_cluster = np.mean(points_of_cluster, axis=0)
    allError = abs(points_of_cluster-centroid_of_cluster)
    maxHSVerr = np.max(allError,axis=0)
    # print(centroid_of_cluster,maxHSVerr)
    # replace color find in clustering to background
    maskForground = np.zeros_like(bkgdMask_flat)
    errOfallColorPixel = abs(colorImg_flat-centroid_of_cluster)
    maskForground[errOfallColorPixel[:,0]>maxHSVerr[0]]=255
    maskForground[errOfallColorPixel[:,1]>maxHSVerr[1]]=255
    maskForground[errOfallColorPixel[:,2]>maxHSVerr[2]]=255
    maskForground = maskForground.astype(np.uint8).reshape((bkgdMask.shape))

    return mask, maskForground,centroid_of_cluster,maxHSVerr

def main():
    for parentfolder in maskParentfolder:
        maskfolder = os.path.join(parentfolder,maskfolderkey)
        maskoutput = os.path.join(parentfolder,"mask_AutoChromakey_HSV_seperateThreshold")
        if not os.path.exists(maskoutput):
            os.makedirs(maskoutput)
        files = glob(os.path.join(maskfolder,'*.png'))
        for file in tqdm(files):
            
            try:
                mask = cv2.imread(file,0)

                # calc bbox mask
                bbox = np.zeros_like(mask)
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                # expand bbox for widther range
                boundary = 10
                rmin = rmin-boundary
                rmax = rmax+boundary
                cmin = cmin-boundary
                cmax = cmax+boundary
                bbox[rmin:rmax,cmin:cmax]=255
                # cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.bbox.png"),bbox)

                # crop color
                colorfilename = f"{os.path.basename(file).split('.')[0]}.color.png"
                color = cv2.imread(os.path.join(parentfolder,colorfilename))
                color_in_bbox = color[rmin:rmax,cmin:cmax]
                # cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.bbox_color.png"),color_in_bbox)

                # crop depth
                depthfilename = f"{os.path.basename(file).split('.')[0]}.depth.png"
                depth = cv2.imread(os.path.join(parentfolder,depthfilename),cv2.IMREAD_UNCHANGED)
                depth_in_bbox = depth[rmin:rmax,cmin:cmax]
                # cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.bbox_depth.png"),depth_in_bbox)

                # clustering to foreground background
                cropForegroundMask = depth2Clustering(depth_in_bbox)
                # cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.depthKmeans.png"),cropForegroundMask)

                # remove outliner in background
                refinedMask,foregroundMask_bbox,MeanColor,maxHSVerr = outlinerRemoveDBSCAN(color_in_bbox, cropForegroundMask)
                
                foundColor = color_in_bbox.copy()
                foundColor[:50,:50,None] = MeanColor + maxHSVerr
                foundColor[50:100,50:100,None] = MeanColor - maxHSVerr
                foundColor[100:150,100:150,None] = MeanColor + maxHSVerr
                foundColor[150:200,150:200,None] = MeanColor - maxHSVerr
                foundColor = cv2.cvtColor(foundColor, cv2.COLOR_HSV2BGR)
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.found.png"),foundColor)

                backgroundColor = color_in_bbox.copy()
                backgroundColor[cropForegroundMask>0] = [255,255,255]
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.backgroundColor.png"),backgroundColor)

                backgroundColor = color_in_bbox.copy()
                backgroundColor[foregroundMask_bbox==0] = [255,255,255]
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.foregroundColor.png"),backgroundColor)

                wholeMask = np.zeros_like(depth).astype(np.uint8)
                wholeMask[rmin:rmax,cmin:cmax] = foregroundMask_bbox
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}"),wholeMask)
            except:
                print("An exception occurred")
            

main()