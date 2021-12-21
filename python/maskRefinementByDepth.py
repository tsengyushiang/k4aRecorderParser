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
    r"D:\projects\k4aRecorderParser\c++\k4aMKVparser\x64\Release\output\6\1"
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

    depthRange = np.max(depth_in_bbox_1d_woNotValid)-np.min(depth_in_bbox_1d_woNotValid)
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

def findFloorBboxfrom2Clustering(label):
    '''
        input : label of foreground or background from depth
        output : crop bbox of floor
        process : find longest "all foreground" region from button
    '''
    foregroundPerncentage = np.sum(label,axis=1)/label.shape[1]/255
    threshold = 0.8
    foregroundPerncentage[foregroundPerncentage<threshold]=0
    foregroundPerncentage[foregroundPerncentage>=threshold]=1.0

    # find longest sequence of 1
    # origion : https://stackoverflow.com/questions/38161606/find-the-start-position-of-the-longest-sequence-of-1s
    idx_pairs = np.where(np.diff(np.hstack(([False],foregroundPerncentage==1,[False]))))[0].reshape(-1,2)

    return idx_pairs[-1][0], idx_pairs[-1][1]

def colorFloorClustering(cropColorOnlyFloor):
    '''
        input : color only have floor region (most is floor and color is similar)
        output : mask of none-floor
        process : kmeans biggest number of clustering count is floor
    '''
    cropColorOnlyFloor = cv2.cvtColor(cropColorOnlyFloor, cv2.COLOR_BGR2HLS) # covert to HSV

    pixel_values = cropColorOnlyFloor.reshape((-1, 3))
    pixel_values = np.float32(pixel_values[:,0])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    clusteringCount = []
    for i in range(k):
        clusteringCount.append(labels[labels==i].shape[0])
    
    mostCluster = np.argmax(clusteringCount)
    mask = np.zeros_like(labels)
    mask[labels != mostCluster] = 255
    mask = mask.reshape(cropColorOnlyFloor.shape[0],cropColorOnlyFloor.shape[1])
    
    return mask

def grabCutwInitialForegroundMask(color_in_bbox,depth_in_bbox,cropForegroundMask,boundary=10):
    '''
        input : possible foreground mask
        output : grabcut mask contour, descript label
        process : grabcut, foregroundMask contours with 10px as cv2.GC_PR_FGD, other mask part as cv2.GC_FGD, others is cv2.GC_BGD
    '''
    # color_in_bbox = cv2.cvtColor(color_in_bbox, cv2.COLOR_BGR2HSV) # covert to HSV

    foregroundmask_bbox = depth_in_bbox.astype(np.uint8)
    descriptLabel_bbox = np.ones_like(foregroundmask_bbox) * cv2.GC_BGD
    # descriptLabel_bbox = np.ones_like(foregroundmask_bbox) * cv2.GC_PR_BGD

    descriptLabel_bbox[0:boundary,:]= cv2.GC_BGD
    descriptLabel_bbox[-boundary:-1,:]= cv2.GC_BGD
    descriptLabel_bbox[:,0:boundary]= cv2.GC_BGD
    descriptLabel_bbox[:,-boundary:-1]= cv2.GC_BGD

    descriptLabel_bbox[depth_in_bbox==0] = cv2.GC_PR_FGD
    descriptLabel_bbox[cropForegroundMask>0] = cv2.GC_FGD
    descriptLabel_bbox_inital = descriptLabel_bbox.copy()*85
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    resultLabel, bgdModel, fgdModel = cv2.grabCut(color_in_bbox,descriptLabel_bbox,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)
    resultMask = np.where((resultLabel==cv2.GC_PR_BGD)|(resultLabel==cv2.GC_BGD),0,255).astype('uint8')

    return descriptLabel_bbox_inital, resultMask

def outlinerRemoveDBSCAN(colorImg, bkgdMask):
    
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2HLS) # covert to HSV
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
    maxErr = np.max(((points_of_cluster-centroid_of_cluster)**2).mean(axis=1))
    # print(centroid_of_cluster,maxErr)
    # replace color find in clustering to background
    maskForground = np.zeros_like(bkgdMask_flat)
    errOfallColorPixel = ((colorImg_flat-centroid_of_cluster)**2).mean(axis=1)
    maskForground[errOfallColorPixel>maxErr]=255
    maskForground = maskForground.astype(np.uint8).reshape((bkgdMask.shape))

    return mask, maskForground


def main():
    for parentfolder in maskParentfolder:
        maskfolder = os.path.join(parentfolder,maskfolderkey)
        maskoutput = os.path.join(parentfolder,"mask_grabCutRefinedPRBG_refined")
        if not os.path.exists(maskoutput):
            os.makedirs(maskoutput)
        files = glob(os.path.join(maskfolder,'*.png'))
        for file in tqdm(files):
            
            # try:

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
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.bbox_color.png"),color_in_bbox)

                # crop depth
                depthfilename = f"{os.path.basename(file).split('.')[0]}.depth.png"
                depth = cv2.imread(os.path.join(parentfolder,depthfilename),cv2.IMREAD_UNCHANGED)
                depth_in_bbox = depth[rmin:rmax,cmin:cmax]
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.bbox_depth.png"),depth_in_bbox)

                # clustering to foreground background
                cropForegroundMask = depth2Clustering(depth_in_bbox)
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.depthKmeans.png"),cropForegroundMask)

                refinedMask,foregroundMask_bbox = outlinerRemoveDBSCAN(color_in_bbox, cropForegroundMask)
                cropForegroundMask = refinedMask
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.depthBGrefined.png"),cropForegroundMask)

                # crop floor region by clustering mask
                rmin_f, rmax_f = findFloorBboxfrom2Clustering(cropForegroundMask)
                color_floor_region = color_in_bbox[rmin_f:rmax_f,:]
                # cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.floor.png"),color_floor_region[:,:,0])

                # get refined mask after floor clustering
                floormask = colorFloorClustering(color_floor_region)
                cropForegroundMask[rmin_f:rmax_f,:] = floormask
                cropForegroundMask[depth_in_bbox==0] = 0
                # cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.floorColorKmeans.png"),cropForegroundMask)

                # refined mask by grabcut
                label, grabcutMask = grabCutwInitialForegroundMask(color_in_bbox,depth_in_bbox,cropForegroundMask,boundary=boundary)
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.grabcutlabel.png"),label)
                grabcutResult = color_in_bbox.copy()
                grabcutResult[grabcutMask==0]=[255,255,255]
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}.grabcutResult.png"),grabcutResult)

                wholeMask = np.zeros_like(depth).astype(np.uint8)
                wholeMask[rmin:rmax,cmin:cmax] = grabcutMask
                cv2.imwrite(os.path.join(maskoutput,f"{os.path.basename(file)}"),wholeMask)
            
            # except:
            #     print("An exception occurred")

main()