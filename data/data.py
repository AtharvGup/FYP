"""
    File to load dataset based on user control from main file
"""
from data.BrainNet import BrainDataset, name2path


def LoadData(DATASET_NAME, threshold=0, edge_ratio=0, node_feat_transform='original', augment=False, augmentation_type="feature_noise"):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """

    if DATASET_NAME in name2path.keys():
        return BrainDataset(DATASET_NAME, threshold=threshold, edge_ratio=edge_ratio,
                            node_feat_transform=node_feat_transform, augment=augment, augmentation_type=augmentation_type)
    else:
        from data.TUs import TUsDataset
        return TUsDataset(DATASET_NAME)
