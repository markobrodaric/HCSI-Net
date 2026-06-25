import torch
import config as c

from models.architecture.HCSINet import HCSINet
from models.backbones.Swin.swin_with_intermediate_inputs import build_swin_backbone
from models.backbones.EfficientNet.EffNet_with_intermediate_inputs import build_efficientnet_backbone
from test import test_cross_dataset
from train import train_hcsi


# Build the model
swin_backbone = build_swin_backbone()
efficientnet_backbone = build_efficientnet_backbone()
model = HCSINet(cnn_backbone=efficientnet_backbone, vit_backbone=swin_backbone)



# Train
history = train_hcsi(
    model=model,
    n_epoch=60,
    batch_size=16,
    weight_decay=1e-4,
    learning_rate=3e-5,
    experiment_name="HCSI_train",
    n_weight=3,
    image_size=224,
)

# Test
checkpoint = torch.load("./weights/HCSI_pretrained.pth", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["state_dict"], strict=True)

results = test_cross_dataset(
    model=model,
    datasets="CDF",     # List of testing datasets, or "all" for evaluation on all datasets 
    test_name="hcsi_cross_dataset_eval",
    dataset_paths=c.dataset_paths,
    image_size=c.image_size,
    num_classes=c.num_classes,
)


