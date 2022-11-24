import torch.nn as nn

class Svhn_dropout_Model(nn.Module):
    def __init__(self):
        super(Svhn_dropout_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=(2,2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(32*3*7,11)
        self.fc2 = nn.Linear(32*3*7,11)
        self.fc3 = nn.Linear(32*3*7,11)
        self.fc4 = nn.Linear(32*3*7,11)
        self.fc5 = nn.Linear(32*3*7,11)
        self.fc6 = nn.Linear(37*3*7,11)

    def froward(self,img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0],-1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)

        return c1,c2,c3,c4,c5,c6

model = Svhn_dropout_Model()
