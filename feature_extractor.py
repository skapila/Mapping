import os
from silx.opencl import sift
import numpy as np
import imageio
from silx.image import sift
import shutil, glob, time
from constants import project_dir, devicetype
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
devicetype = "GPU"

class SiftFeatureExtractor:
    def __init__(self,):
        self.project_dir = project_dir
        self.images_dir = os.path.join(self.project_dir)
        self.features_dir = os.path.join(self.project_dir, "features")
        if os.path.exists(self.features_dir):
            shutil.rmtree(self.features_dir)
        os.makedirs(self.features_dir)
        self.images_list = glob.glob(os.path.join(self.images_dir, "*.JPG"))
        print(f"{devicetype=}")
        self.sift_ocl = sift.SiftPlan(template=imageio.imread(self.images_list[0]), devicetype=devicetype)

    def feature_extraction(self, image_path):
        keypoints = self.sift_ocl.keypoints(imageio.imread(image_path))
        return keypoints

    def save_feature(self, image_path, keypoints):
        image_name = os.path.basename(image_path)
        print("saving feature")
        np.save(file=os.path.join(self.features_dir, f"{image_name}.npy"), arr=keypoints)


