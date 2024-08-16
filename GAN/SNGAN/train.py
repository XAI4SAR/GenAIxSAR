import sys

sys.path.append("./acgan_code")
import os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import mstar_dataset
import math
import torch.autograd as autograd
import numpy as np
from net import Discriminator, Generator
from tensorboardX import SummaryWriter

writer = SummaryWriter('./acgan_code/tensorboard/' + 'ACGAN')




# training parameters
batch_size = 32

lr_G = 0.0001
lr_D = 0.0001
train_epoch = 500

# data_loader


data_transforms = transforms.Compose([
    transforms.ToTensor(),
])
dataset_train = mstar_dataset.MSTAR_Dataset(txt_file='./MSTAR_10CLASS/src/train_c2.txt',
                                            transform=data_transforms,
                                            )
# dataloader_train = DataLoader(dataset_train, batch_size=32,shuffle=True,num_workers=8)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = mstar_dataset.MSTAR_Dataset(txt_file='./MSTAR_10CLASS/src/test_c2.txt',
                                           transform=data_transforms,
                                           )
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=8)

# network
# G = generator(128)
# D = discriminator(128)
G = Generator()
D = Discriminator()
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
Decision_loss = nn.MSELoss()
Class_loss = nn.CrossEntropyLoss()
Az_loss = nn.MSELoss()
one2onemapping = nn.L1Loss()
# Az_loss = nn.L1Loss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

# fixed noise & label&az
# fixed_z_ = torch.randn(100, 64).view(-1, 64, 1, 1)
fixed_z_ = torch.randn(100, 64)
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
# fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)

fixed_az = torch.deg2rad(360 * torch.rand(100, 1))
# fixed_az = 360*torch.rand(100, 1)/180*math.pi
fixed_az_vec = torch.zeros(100, 10)
for i in range(100):
    fixed_az_vec[i][0] = torch.cos(fixed_az[i])
    fixed_az_vec[i][1] = torch.sin(fixed_az[i])
    fixed_az_vec[i][2] = torch.cos(2 * fixed_az[i])
    fixed_az_vec[i][3] = torch.sin(2 * fixed_az[i])
    fixed_az_vec[i][4] = torch.cos(3 * fixed_az[i])
    fixed_az_vec[i][5] = torch.sin(3 * fixed_az[i])
    fixed_az_vec[i][6] = torch.cos(4 * fixed_az[i])
    fixed_az_vec[i][7] = torch.sin(4 * fixed_az[i])
    fixed_az_vec[i][8] = torch.cos(5 * fixed_az[i])
    fixed_az_vec[i][9] = torch.sin(5 * fixed_az[i])
# fixed_az_vec = fixed_az_vec.view(-1, 2, 1, 1)
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())
    fixed_y_label_ = Variable(fixed_y_label_.cuda())
    fixed_az_vec = Variable(fixed_az_vec.cuda())


def show_result(num_epoch, show=False, save=False, path='result.png'):  #

    G.eval()
    test_images = G(fixed_z_, fixed_y_label_, fixed_az_vec)

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10 * 10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# results save folder
root = 'MSTAR/'
model = 'MSTAR_ACGAN_LSGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['C_G_losses'] = []
train_hist['A_G_losses'] = []
train_hist['C_D_losses'] = []
train_hist['A_D_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1)

print('training start!')
start_time = time.time()

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    C_G_losses = []
    A_G_losses = []
    C_D_losses = []
    A_D_losses = []

    # # learning rate decay

    if (epoch + 1) <= 80:
        d_steps = 1
        g_steps = 5
    else:
        d_steps = 1
        g_steps = 1

    epoch_start_time = time.time()

    acc_num_g = 0.0
    data_num_g = 0.0

    for idx, data in enumerate(train_loader):
        x_data = data['image']
        x_label = data['label'].cuda()
        x_az = data['az']
        D.zero_grad()

        mini_batch = x_data.size()[0]

        y_real = torch.ones(mini_batch)
        y_fake = torch.zeros(mini_batch)
        y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())

        real_label = onehot[x_label]

        random_z = torch.randn((mini_batch, 64))

        real_angle = torch.deg2rad(x_az)
        real_az_vec = torch.zeros(mini_batch, 10)
        for i in range(mini_batch):
            real_az_vec[i][0] = torch.cos(real_angle[i])
            real_az_vec[i][1] = torch.sin(real_angle[i])
            real_az_vec[i][2] = torch.cos(2 * real_angle[i])
            real_az_vec[i][3] = torch.sin(2 * real_angle[i])
            real_az_vec[i][4] = torch.cos(3 * real_angle[i])
            real_az_vec[i][5] = torch.sin(3 * real_angle[i])
            real_az_vec[i][6] = torch.cos(4 * real_angle[i])
            real_az_vec[i][7] = torch.sin(4 * real_angle[i])
            real_az_vec[i][8] = torch.cos(5 * real_angle[i])
            real_az_vec[i][9] = torch.sin(5 * real_angle[i])

        x_data = Variable(x_data.cuda())
        real_label = Variable(real_label.cuda())
        real_az_vec = Variable(real_az_vec.cuda())
        random_z = Variable(random_z.cuda())
        x_az = Variable(x_az.cuda())

        for d_index in range(d_steps):
            # train discriminator D
            d_real, class_real, theta_real = D(x_data)
            decison_real = d_real.squeeze()
            class_real = class_real.squeeze()
            theta_real = theta_real.squeeze()

            D_real_loss = Decision_loss(decison_real, y_real)
            C_real_loss = Class_loss(class_real, real_label)
            A_real_loss = Az_loss(theta_real, real_az_vec)

            x_re = G(random_z, real_label, real_az_vec)
            d_fake, class_fake, theta_fake = D(x_re)
            decison_fake = d_fake.squeeze()
            class_fake = class_fake.squeeze()
            theta_fake = theta_fake.squeeze()

            D_fake_loss = Decision_loss(decison_fake, y_fake)
            C_fake_loss = Class_loss(class_fake, real_label)
            A_fake_loss = Az_loss(theta_fake, real_az_vec)

            D_train_loss = (D_fake_loss + D_real_loss ) + 5 * (C_real_loss + C_fake_loss) + 20 * (A_real_loss + A_fake_loss)

            C_Dtrain_losses = C_real_loss
            A_Dtrain_losses = A_real_loss
            D_train_loss.backward()
            D_optimizer.step()

        D_losses.append(D_fake_loss - D_real_loss)
        C_D_losses.append(C_Dtrain_losses)
        A_D_losses.append(A_Dtrain_losses)

        for g_index in range(g_steps):
            # train generator G
            G.zero_grad()

            x_re = G(random_z, real_label, real_az_vec)
            d_fake, class_fake, theta_fake = D(x_re)
            decison_fake = d_fake.squeeze()
            class_fake = class_fake.squeeze()
            theta_fake = theta_fake.squeeze()

            D_fake_loss = Decision_loss(decison_fake, y_real)
            C_Gtrain_loss = Class_loss(class_fake, real_label)
            A_Gtrain_loss = Az_loss(theta_fake, real_az_vec)

            G_train_loss = (D_fake_loss + 5 * C_Gtrain_loss + 20 * A_Gtrain_loss + one2onemapping(x_re, x_data))
            G_train_loss.backward()
            G_optimizer.step()

        G_losses.append(D_fake_loss)
        C_G_losses.append(C_Gtrain_loss)
        A_G_losses.append(A_Gtrain_loss)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print(torch.mean(torch.FloatTensor(A_D_losses)))
    print(torch.mean(torch.FloatTensor(A_G_losses)))

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f  ' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['C_G_losses'].append(torch.mean(torch.FloatTensor(C_G_losses)))
    train_hist['C_D_losses'].append(torch.mean(torch.FloatTensor(C_D_losses)))
    train_hist['A_G_losses'].append(torch.mean(torch.FloatTensor(A_G_losses)))
    train_hist['A_D_losses'].append(torch.mean(torch.FloatTensor(A_D_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    writer.add_scalars('G_losses', {'train': torch.mean(torch.FloatTensor(G_losses))}, epoch + 1)
    writer.add_scalars('D_losses', {'train': torch.mean(torch.FloatTensor(D_losses))}, epoch + 1)
    writer.add_scalars('C_losses',
                       {'D': torch.mean(torch.FloatTensor(C_D_losses)), 'G': torch.mean(torch.FloatTensor(C_G_losses))},
                       epoch + 1)
    writer.add_scalars('A_losses',
                       {'D': torch.mean(torch.FloatTensor(A_D_losses)), 'G': torch.mean(torch.FloatTensor(A_G_losses))},
                       epoch + 1)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')  # 保存训练好的模型
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')  # 保存训练好的模型
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)


