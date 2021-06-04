from fastai.vision.all import *
from jeremai.keypoints.transforms import RandomBinaryHeatmapTransform, HeatmapPoint
from jeremai.keypoints.metrics import binary_heatmap_loss, nmae_topk
from jeremai.keypoints.models import hrnetmodel
from jeremai.keypoints.datablock import get_y

class Predictor:
    learner = None

    def __init__(self, fname):
        self.learner = load_learner(fname)

    def predict(self, im_pth):
        im, ratio = self.prepareImageForPrediction(im_pth)
        keypoints, h1, h2 = self.learner.predict(im)
        keypoints = keypoints/ratio

        return [kp.numpy() for kp in keypoints]

    def prepareImageForPrediction(self, im_pth, desired_size=256):

        im = Image.open(im_pth)
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        im = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, (0,0))

        return np.array(new_im), ratio    
