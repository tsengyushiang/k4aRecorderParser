from glob import glob
from tqdm import tqdm
import os
import cv2
import numpy as np 

def VisIOInSameFrame(imagedis):
    for folder in imagedis:
        files = glob(os.path.join(folder,'*.color.png'))        
        maskfolder = os.path.join(folder,'mask_AutoChromakey')
        refinedmaskfolder = os.path.join(folder,'mask_grabCutRefinedPRBG')

        outputVideo = os.path.join(folder,'compare_PRBG.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        for file in tqdm(files):
            maskfile = os.path.join(maskfolder,os.path.basename(file))
            refinedmaskfile = os.path.join(refinedmaskfolder,os.path.basename(file))
            if not os.path.exists(maskfile):
                continue
            elif not os.path.exists(maskfile):
                continue
            else:
                input=cv2.imread(file)
                mask=cv2.imread(maskfile,0)
                refinedmask=cv2.imread(refinedmaskfile,0)
                bgremoved = input.copy()
                bgremoved[mask==0] = [255,255,255]
                refinedbgremoved = input.copy()
                refinedbgremoved[refinedmask==0] = [255,255,255]

                compared = np.vstack((input,refinedbgremoved))
                scale_percent = 50 # percent of original size
                width = int(compared.shape[1] * scale_percent / 100)
                height = int(compared.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(compared, dim)

                if out is None:
                    out = cv2.VideoWriter(outputVideo, fourcc, 30, (resized.shape[1], resized.shape[0]))

                out.write(resized)
                # cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
                # cv2.imshow('My Image', compared)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        
        out.release()

from argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')   
    args.add_argument('-h', '--help', action='help',default='SUPPRESS')
    # custom command line input parameters       
    args.add_argument("--folder",type=str,default=None)
    return parser

args = build_argparser().parse_args()

if args.folder:
    VisIOInSameFrame([args.folder])
else:
    print('use --folder to specify process folder.')