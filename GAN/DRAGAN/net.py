import torch
import torch.nn as nn
import torch.nn.functional as F
#128
class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # 2fc
        self.g_layer1 = nn.ConvTranspose2d(84, 1024, 4, 1, 0)
        self.g_layer2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.g_layer3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.g_layer4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.g_layer5 = nn.ConvTranspose2d(128, 32, 4, 2, 1)
        self.g_layer6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label, az):
        x = torch.cat([input, label, az], 1)
        x = x.view(-1, 84, 1, 1)
        x = self.g_layer1(x)
        x = self.LeakyReLU(x)

        x = self.g_layer2(x)
        x = self.LeakyReLU(x)

        x = self.g_layer3(x)
        x = self.LeakyReLU(x)

        x = self.g_layer4(x)
        x = self.LeakyReLU(x)

        x = self.g_layer5(x)
        x = self.LeakyReLU(x)

        x = self.g_layer6(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d_layer1 = nn.utils.spectral_norm(nn.Conv2d(1, 64, 4, 2, 1))
        self.d_layer2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.d_layer3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.d_layer4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1))
        self.d_layer5 = nn.utils.spectral_norm(nn.Conv2d(512, 1024, 8, 1, 0))

        self.d_finial_layer = nn.utils.spectral_norm(nn.Conv2d(1024, 1, 1, 1, 0))
        self.c_layer1 = nn.utils.spectral_norm(nn.Conv2d(1024, 256, 1, 1, 0))
        self.c_layer2 = nn.utils.spectral_norm(nn.Conv2d(256, 10, 1, 1, 0))

        self.az_layer1 = nn.utils.spectral_norm(nn.Conv2d(1024, 256, 1, 1, 0))
        self.az_layer2 = nn.utils.spectral_norm(nn.Conv2d(256, 10, 1, 1, 0))

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.d_layer1(input)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.d_layer2(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.d_layer3(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.d_layer4(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.d_layer5(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        d = self.d_finial_layer(x)
        d = self.sigmoid(d)

        c = self.c_layer1(x)
        c = self.LeakyReLU(c)
        c = self.c_layer2(c)
        c = self.softmax(c)

        a = self.az_layer1(x)
        a = self.LeakyReLU(a)
        a = self.az_layer2(a)
        a = self.tanh(a)
        return d, c, a




def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()