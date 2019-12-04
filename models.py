from torch import nn
from torchsummary import summary
from torchvision import models

from config import device


class FaceAttributeModel(nn.Module):
    def __init__(self):
        super(FaceAttributeModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 17)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        reg = self.sigmoid(x[:, :5])  # [N, 8]
        expression = self.softmax(x[:, 5:8])
        gender = self.softmax(x[:, 8:10])
        glasses = self.softmax(x[:, 10:13])
        race = self.softmax(x[:, 13:17])
        return reg, expression, gender, glasses, race


if __name__ == "__main__":
    model = FaceAttributeModel().to(device)
    summary(model, input_size=(3, 224, 224))

