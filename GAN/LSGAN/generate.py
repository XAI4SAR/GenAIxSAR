import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch.utils.data.dataloader as DataLoader
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import mstar_dataset
from  torchvision import utils as vutils
from torch.autograd import Variable
from net import Discriminator,Generator


G = Generator()
G.cuda()
G.load_state_dict(torch.load('./MSTAR/MSTAR_ACGAN_LSGAN_generator_param.pkl')) #加载预训练好的模型权重


batch_size = 32
data_transforms = transforms.Compose([
        transforms.ToTensor(),
])

dataset_test = mstar_dataset.MSTAR_Dataset(txt_file='../data/test.txt',
                                            transform=data_transforms,
                                            )
# dataloader_train = DataLoader(dataset_train, batch_size=32,shuffle=True,num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size, shuffle=True)

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)

ssim_all = 0.0
data_num = 0.0
all_errors = 0
num = 0

root_directory = './MSTAR/generated'
# 检查根目录是否存在，如果不存在，则创建它
if not os.path.exists(root_directory):
    os.makedirs(root_directory)
# 创建以数字 0 到 9 命名的文件夹
for i in range(10):
    folder_name = str(i)
    folder_path = os.path.join(root_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for idx, data in enumerate(test_loader):
    x_data = data['image']
    x_label = data['label']
    x_az = data['az']
    x_name = data['name']
    mini_batch = x_data.size()[0]
   

    real_label = onehot[x_label]
    random_z = torch.randn((mini_batch, 64))

    real_angle = torch.deg2rad(x_az)
    real_az_vec = torch.zeros(mini_batch, 10)
    for i in range(mini_batch):
        real_az_vec[i][0] = torch.cos(real_angle[i])
        real_az_vec[i][1] = torch.sin(real_angle[i])
        real_az_vec[i][2] = torch.cos(2*real_angle[i])
        real_az_vec[i][3] = torch.sin(2*real_angle[i])
        real_az_vec[i][4] = torch.cos(3*real_angle[i])
        real_az_vec[i][5] = torch.sin(3*real_angle[i])
        real_az_vec[i][6] = torch.cos(4*real_angle[i])
        real_az_vec[i][7] = torch.sin(4*real_angle[i])
        real_az_vec[i][8] = torch.cos(5*real_angle[i])
        real_az_vec[i][9] = torch.sin(5*real_angle[i])      
        
    x_data = Variable(x_data.cuda())
    real_label = Variable(real_label.cuda())
    real_az_vec = Variable(real_az_vec.cuda())
    random_z = Variable(random_z.cuda())
    x_az = Variable(x_az.cuda())

    x_re = G(random_z, real_label, real_az_vec)

    for i in range(mini_batch):
        vutils.save_image(x_re[i], './MSTAR/generated/{}/{}_{}.jpg'.format(x_label[i],x_name[i],x_az[i]), normalize=False,cmap='gray')
    