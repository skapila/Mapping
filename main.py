from feature_extractor import SiftFeatureExtractor
import glob, os
from constants import project_dir

fx = SiftFeatureExtractor()
images = glob.glob(os.path.join(project_dir,"*.JPG"))
for p in images:
    kp = fx.feature_extraction(p)
    fx.save_feature(p, kp)

