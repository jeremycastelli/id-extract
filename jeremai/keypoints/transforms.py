from fastai.vision.all import *
from jeremai.keypoints.utils.gaussian import generate_gaussian

def _scale(p, s): return 2 * (p / s) - 1

def heatmap2argmax(heatmap, scale=False):
    N, C, H, W = heatmap.shape
    index = heatmap.view(N,C,1,-1).argmax(dim=-1)
    pts = torch.cat([index%W, index//W], dim=2)
    
    if scale:
        scale = tensor([W,H], device=heatmap.device)
        pts = _scale(pts, scale)
    
    return pts

class Heatmap(TensorImageBase): 
    "Heatmap tensor, we can use the type to modify how we display things"
    pass

class HeatmapPoint(TensorPoint):
    """
    A class that mimics TensorPoint, but wraps it so
    we'll be able to override `show` methods with
    a different type.
    """
    pass

class HeatmapTransform(Transform):
    """
    A batch transform that turns TensorPoint instances into Heatmap instances,
    and Heatmap instances into HeatmapPoint instances.
    
    Used as the last transform after all other transformations. 
    """
    # We want the heat map transformation to happen last, so give it a high order value
    order=999
    
    def __init__(self, heatmap_size, sigma=10, **kwargs):
        """
        heatmap_size: Size of the heatmap to be created
        sigma: Standard deviation of the Gaussian kernel
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.size = heatmap_size
    
    def encodes(self, x:TensorPoint):
        # The shape of x is (batch x n_points x 2)
        num_imgs = x.shape[0]
        num_points = x.shape[1]
        
        maps = Heatmap(torch.zeros(num_imgs, num_points, *self.size, device=x.device))
        for b,c in itertools.product(range(num_imgs), range(num_points)):
            # Note that our point is already scaled to (-1, 1) by PointScaler
            point = x[b][c]
            generate_gaussian(maps[b][c], point[0], point[1], sigma=self.sigma)
        
        return maps
    
    def decodes(self, x:Heatmap):
        """
        Decodes a heat map back into a set of points by finding
        the coordinates of their argmax.
        
        This returns a HeatmapPoint class rather than a TensorPoint
        class, so we can modify how we display the output.
        """
        # Flatten the points along the final axis,
        # and find the argmax per channel
        xy = heatmap2argmax(x, scale=True)
        return HeatmapPoint(xy, source_heatmap=x)

def coord2heatmap(x, y, w, h, heatmap):
    """
    Inserts a coordinate (x,y) from a picture with 
    original size (w x h) into a heatmap, by randomly assigning 
    it to one of its nearest neighbor coordinates, with a probability
    proportional to the coordinate error.
    
    Arguments:
    x: x coordinate
    y: y coordinate
    w: original width of picture with x coordinate
    h: original height of picture with y coordinate
    """
    # Get scale
    oh,ow = heatmap.shape
    sx = ow / w
    sy = oh / h
    
    # Unrounded target points
    px = x * sx
    py = y * sy
    
    # Truncated coordinates
    nx,ny = int(px), int(py)
    
    # Coordinate error
    ex,ey = px - nx, py - ny
    
    xyr = torch.rand(2, device=heatmap.device)
    xx = (ex >= xyr[0]).long()
    yy = (ey >= xyr[1]).long()
    heatmap[min(ny + yy, heatmap.shape[0] - 1), 
            min(nx+xx, heatmap.shape[1] - 1)] = 1
    
    return heatmap

def heatmap2coord(heatmap, topk=9):
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N,C,1,-1).topk(topk, dim=-1)
    coord = torch.cat([index%W, index//W], dim=2)
    return (coord*F.softmax(score, dim=-1)).sum(-1)

def topk_heatmap2coord(heatmap, topk=9, scale=False):
    coord = heatmap2coord(heatmap, topk)
    if scale:
        _, _, H, W = heatmap.shape
        scale = tensor([W,H], device=heatmap.device)
        coord = _scale(coord, scale)
    
    return coord

class RandomBinaryHeatmapTransform(Transform):
    order=999
    
    def __init__(self, heatmap_size, topk=9, **kwargs):
        super().__init__(**kwargs)
        self.size = tensor(heatmap_size)
        self.topk=topk
    
    def encodes(self, x:TensorPoint):
        # The shape of x is (batch x n_points x 2)
        num_imgs = x.shape[0]
        num_points = x.shape[1]
        
        maps = Heatmap(torch.zeros(num_imgs, num_points, *self.size, dtype=torch.long,
                                   device=x.device))
        for b,c in itertools.product(range(num_imgs), range(num_points)):
            heatmap = maps[b][c]
            
            # Note that our point is already scaled to (-1, 1) by PointScaler.
            # We pretend here it's in range 0...2
            point = x[b][c] + 1.
            coord2heatmap(point[0], point[1], 2., 2., heatmap)
        
        return maps
    
    def decodes(self, x:Heatmap):
        """
        Decodes a batch of binary heatmaps back into a set of
        TensorPoints.
        """
        if x.dtype == torch.long:
            # If the target heatmap is an annotation heatmap, our
            # decoding procedure is different - we need to directly
            # retrieve the argmax.
            return HeatmapPoint(heatmap2argmax(x, scale=True),
                               source_heatmap=x)
        
        return HeatmapPoint(topk_heatmap2coord(x, topk=self.topk, scale=True),
                           source_heatmap=x)