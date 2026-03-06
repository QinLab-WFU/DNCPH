import torch
import torch.nn.functional as F
import torchvision
from torch import nn


def build_model(args, pretrained=True):

    allowed_items = {"resnet50", "frozen", "double", "normalize", "layernorm"}
    if "resnet50" in args.backbone and all(item in allowed_items for item in args.backbone.split("_")):
        net = ResNet50(
            args.n_bits,
            pretrained,
            frozen="frozen" in args.backbone,
            double="double" in args.backbone,
            normalize="normalize" in args.backbone,
            layernorm="layernorm" in args.backbone,
        )
    else:
        raise NotImplementedError(f"not support: {args.backbone}")
    return net.cuda()


class ResNet50(nn.Module):
    def __init__(self, n_bits, pretrained=True, **kwargs):
        super().__init__()

        self.frozen = kwargs.pop("frozen", False)
        self.double = kwargs.pop("double", False)
        self.normalize = kwargs.pop("normalize", False)
        self.layernorm = kwargs.pop("layernorm", False)

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)
        self.dim_feature = self.model.fc.in_features

        # self.models.fc = nn.Linear(self.dim_feature, n_bits, bias=False)
        self.model.fc = nn.Linear(self.dim_feature, n_bits)
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)
        # nn.init.kaiming_normal_(self.models.fc.weight, mode="fan_out")
        # nn.init.constant_(self.models.fc.bias, 0)

        if self.double:
            self.model.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        if self.frozen:
            for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.model.modules()):
                module.eval()
                module.train = lambda _: None

        if self.layernorm:
            self.layer_norm = nn.LayerNorm(n_bits, elementwise_affine=False)

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        if self.double:
            x = self.model.avgpool(x) + self.model.maxpool2(x)
        else:
            x = self.model.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        if self.layernorm:
            x = self.layer_norm(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x


if __name__ == "__main__":
    pass
