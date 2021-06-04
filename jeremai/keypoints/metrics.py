from fastai.vision.all import *
from jeremai.keypoints.transforms import heatmap2argmax, topk_heatmap2coord

def nmae_argmax(preds, targs):
    # Note that our function is passed two heat maps, which we'll have to
    # decode to get our points. Adding one and dividing by 2 puts us
    # in the range 0...1 so we don't have to rescale for our percentual change.
    preds = 0.5 * (TensorBase(heatmap2argmax(preds, scale=True)) + 1)
    targs = 0.5 * (TensorBase(heatmap2argmax(targs, scale=True)) + 1)
    
    return ((preds-targs).abs()).mean()

def gaussian_heatmap_loss(pred, targs):
    targs = TensorBase(targs)
    pred = TensorBase(pred)
    return F.mse_loss(pred, targs)

def binary_heatmap_loss(preds, targs, pos_weight=None, topk=9):
    preds = TensorBase(preds)
    targs = TensorBase(targs).float()
    
    if pos_weight is not None:
        _,p,h,w=preds.shape
        pos_weight=torch.tensor(pos_weight, device=preds.device).expand(p, h, w)
    
    return F.binary_cross_entropy_with_logits(preds, targs, pos_weight=pos_weight)    

def nmae_topk(preds, targs, topk=9):
    # Note that our function is passed two heat maps, which we'll have to
    # decode to get our points. Adding one and dividing by 2 puts us
    # in the range 0...1 so we don't have to rescale for our percentual change.
    preds = 0.5 * (TensorBase(topk_heatmap2coord(preds, topk=topk, scale=True)) + 1)
    targs = 0.5 * (TensorBase(heatmap2argmax(targs, scale=True)) + 1)
    
    return ((preds-targs).abs()).mean()
