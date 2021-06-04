from jeremai.keypoints.hrnet import hrnet18
from fastai.vision.all import nn

class BinaryHeadBlock(nn.Module):
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, out_channels, 1, bias=False),
        )
        
    def forward(self, input):
        return self.layers(input)

def hrnetmodel(nb_points):
    backbone = hrnet18()
    head = BinaryHeadBlock(270, 270, nb_points)
    model = nn.Sequential(backbone, head)
    return model