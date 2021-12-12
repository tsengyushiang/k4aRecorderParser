"""
### Windows demo installation instuction 

```
conda create -n detectron python=3.8
conda activate detectron

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install cython

D:

git clone https://github.com/facebookresearch/detectron2.git

cd detectron2
pip install -e .
pip install opencv-python
```

### Error

- `ImportError: DLL load failed while importing win32file: 找不到指定的程序。`
    ```
    conda install pywin32
    ```
"""
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import detectron2
from detectron2.projects import point_rend

import cv2
import numpy as np

class Detector:

    def __init__(self,model_type):
        self.cfg = get_cfg()

        #Load model config and pretrained model
        if model_type=="mask_rcnn":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type=="pointrend":
            point_rend.add_pointrend_config(self.cfg)

            # find yaml where you clone detecton2 from git
            self.cfg.merge_from_file(r"D:\projects\detectron2\projects\PointRend\configs\InstanceSegmentation\pointrend_rcnn_R_50_FPN_3x_coco.yaml")
            # copy from right clik PretrainedModels/InstanceSegmentation/COCO/download hyperlink : model
            self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
        self.cfg.MODEL.DEVICE="cpu" #cpu or cuda       

        self.predictor = DefaultPredictor(self.cfg)
    
    def cropper(self,img, mask_array,class_array,class_dict,mask=[]):
        num_instances = mask_array.shape[0]
        mask_array = np.moveaxis(mask_array, 0, -1)
        output = np.zeros_like(img)
        for i in range(num_instances):            
            if class_dict[class_array[i]] in mask:
                output = np.where(mask_array[:, :, i:(i+1)] == True, 255, output)
        return output

    def onImage(self,imagePath):
        image=cv2.imread(imagePath)
        predictions = self.predictor(image)

        catalog = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        vis = Visualizer(image[:,:,::-1],metadata=catalog,scale=1.2)

        output =vis.draw_instance_predictions(predictions["instances"].to("cpu"))
        labeledImg = output.get_image()[:, :, ::-1]

        print('possible class :',catalog.thing_classes)
        mask = self.cropper(
            image,
            predictions["instances"].pred_masks.numpy(),
            predictions["instances"].pred_classes.numpy(),
            catalog.thing_classes,
            ["person"]
        )

        return mask, labeledImg

from glob import glob
import os

def MaskPredFromRGB(imagedis):
    dectector = Detector(model_type="pointrend")
    for folder in imagedis:
        files = glob(os.path.join(folder,'*.color.png'))
        
        outputfolder = os.path.join(folder,'mask')
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
        labelfolder = os.path.join(folder,'label')
        if not os.path.exists(labelfolder):
            os.mkdir(labelfolder)
        for file in files:
            outputfile = os.path.join(outputfolder,os.path.basename(file))
            labelfile = os.path.join(labelfolder,os.path.basename(file))
            print(file,outputfile)
            mask,labeledImg = dectector.onImage(file)
            cv2.imwrite(outputfile,mask)
            cv2.imwrite(labelfile,labeledImg)

def VisIOInSameFrame(imagedis):
    for folder in imagedis:
        files = glob(os.path.join(folder,'*.png'))        
        outputfolder = os.path.join(folder,'mask')

        outputVideo = os.path.join(folder,'compare.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        print(folder)
        for file in files:
            outputfile = os.path.join(outputfolder,os.path.basename(file))
            if not os.path.exists(outputfile):
                continue
            else:
                input=cv2.imread(file)
                mask=cv2.imread(outputfile)
                bgremoved = cv2.bitwise_and(input, mask)

                compared = np.vstack((input,bgremoved))
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
    MaskPredFromRGB([args.folder])
else:
    print('use --folder to specify process folder.')