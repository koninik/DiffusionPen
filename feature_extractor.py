import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=0, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=num_classes, global_pool="max"
        )
        #self.model = torch.compile(self.model, backend="inductor")
        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        x = self.model(x)
        return x   