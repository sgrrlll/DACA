from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x,SimpleVoxelBackBone8x,VoxelResBackBone8x
from .votr_backbone import VoxelTransformer, VoxelTransformerV2, VoxelTransformerV3
__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'SimpleVoxelBackBone8x': SimpleVoxelBackBone8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelTransformer': VoxelTransformer,
    'VoxelTransformerV3': VoxelTransformerV3,
}

################下面是我加的
# from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
# from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
# from .spconv_backbone_pruning import VoxelPruningResBackBone8x_SPSS_SPRS, VoxelPruningBackBone8x_SPSS_SPRS
# from .spconv_unet import UNetV2
# from .spconv_backbone_pruning import *

# __all__ = {
#     'VoxelBackBone8x': VoxelBackBone8x,
#     'UNetV2': UNetV2,
#     'PointNet2Backbone': PointNet2Backbone,
#     'PointNet2MSG': PointNet2MSG,
#     'VoxelResBackBone8x': VoxelResBackBone8x,
#     'VoxelPruningResBackBone8x_SPSS_SPRS': VoxelPruningResBackBone8x_SPSS_SPRS, 
#     'VoxelPruningBackBone8x_SPSS_SPRS': VoxelPruningBackBone8x_SPSS_SPRS
# }