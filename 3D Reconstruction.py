import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh
from os import listdir
import cv2
import pyrender
import matplotlib.pyplot as plt


class ResidualLayer(nn.Module):
    '''
        Used as a basic residual module for a ResidualBlock.
        Pre-activation scheme Used: BN-ReLU-Conv

        Parameters:
            in_channels - {int} - number of input channels
            out_channels - {int} - number of output channels
            stride - {int} - stride to use in Convolution layers
                                (if stride > 1, the layer is used as a bottleneck layer)
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out


class ResidualBlock(nn.Module):
    '''
        ResidualBlock comprised of 3 residual modules,
        all utilizing (3 × 3) kernels. The first residual module
        in each block reduces the spatial resolution by 2 via strided
        convolutions and increases the number of feature maps by 2.

        Parameters:
            in_channels - {int} - number of input channels
            out_channels - {int} - number of output channels
    '''

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.layer1 = ResidualLayer(in_channels, out_channels, 2)
        self.layer2 = ResidualLayer(out_channels, out_channels, 1)
        self.layer3 = ResidualLayer(out_channels, out_channels, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class ListModule(object):
    def __init__(self, module, prefix, *modules):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in modules:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class FeatureToWeights(nn.Module):
    '''
        Used as a final Layer in the ResNet model -
             splits the ResNet network into multiple heads. There is
                        one group of heads per each layer of MeshNet.

        Parameters:
            in_channels - {int} - number of input channels
            out_channels - {int} - number of output channels
            hidden_layers - {int} - number of hidden layers used in MeshNet
    '''

    def __init__(self, input_size, output_size, hidden_layers=4):
        super(FeatureToWeights, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size

        self.W = ListModule(self, 'W')
        self.B = ListModule(self, 'B')
        self.S = ListModule(self, 'S')

        for i in range(1, self.hidden_layers + 2):
            if i == 1:
                self.W.append(nn.Linear(input_size, 3 * 32))
                self.B.append(nn.Linear(input_size, 1 * 32))
                self.S.append(nn.Linear(input_size, 1 * 32))
            elif i < self.hidden_layers + 1:
                self.W.append(nn.Linear(input_size, 32 * 32))
                self.B.append(nn.Linear(input_size, 1 * 32))
                self.S.append(nn.Linear(input_size, 1 * 32))
            else:
                self.W.append(nn.Linear(input_size, 32 * output_size))
                self.B.append(nn.Linear(input_size, 1 * output_size))
                self.S.append(nn.Linear(input_size, 1 * output_size))

    def forward(self, x):
        out = {}
        i = 1
        for W, B, S in zip(self.W, self.B, self.S):
            if i == 1:
                out[f'W{i}'] = W(x).reshape((32, 3))
            elif i < self.hidden_layers + 1:
                out[f'W{i}'] = W(x).reshape((32, 32))
            else:
                out[f'W{i}'] = W(x).reshape(self.output_size, 32)
            out[f'B{i}'] = B(x)
            out[f'S{i}'] = S(x)
            i += 1
        return out


class ResNet(nn.Module):
    def __init__(self, output_size, N=64, hidden_layers=4):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, N, kernel_size=5, stride=1, padding=2)
        self.block1 = ResidualBlock(N * 1, N * 2)
        self.block2 = ResidualBlock(N * 2, N * 4)
        self.block3 = ResidualBlock(N * 4, N * 8)
        self.block4 = ResidualBlock(N * 8, N * 16)
        self.block5 = ResidualBlock(N * 16, N * 32)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, N * 16))
        self.fc1 = nn.Linear(N * 16, N * 16)
        self.fc2 = nn.Linear(N * 16, N * 16)
        self.features_to_weights = FeatureToWeights(N * 16, output_size, hidden_layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.features_to_weights(out)
        return out


class LinearLayer(nn.Module):
    '''
        Used as a basic Linear module for a MeshNet.

        Parameters:
            weight - {tensor} - weight matrix
            bias - {tensor} - bias vector
            scale - {tensor} - scale vector
    '''

    def __init__(self, weight, bias, scale, activation='elu'):
        super(LinearLayer, self).__init__()
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax()
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        self.bias = torch.nn.Parameter(bias, requires_grad=True)
        self.scale = torch.nn.Parameter(scale, requires_grad=True)

    def forward(self, x):
        out = torch.mm(x, self.weight.t())
        out = out * self.scale
        out = out + self.bias
        out = self.activation(out)
        return out


class MeshNet(nn.Module):
    '''
        Predicts Whether a 3D point is inside the Mesh using the output from
        the ResNet model as weights.

        Functions:
            create_layers:
                initializes all layers fo MeshNet using predicted weights the ResNet model.
                Parameters:
                    weights - {dict} - containing al the weights.
    '''

    def __init__(self):
        super(MeshNet, self).__init__()
        self.activation = nn.Softmax(dim=0)  # inplace=True)

    def forward(self, x, weights):
        # self.weights = weights
        out = x
        for i in range(1, int(len(weights) / 3) + 1):
            out = torch.mm(out, weights[f"W{i}"].t())
            out = out * weights[f"S{i}"]
            out = out + weights[f"B{i}"]
            if i == int(len(weights) / 3):
                out = nn.Softmax(dim=0)(out)
            else:
                out = nn.ELU()(out)
        return out


class Model(nn.Module):
    '''
        End-To-End Model (Resnet --> MeshNet).
        uses SGD optimizer and BCELoss.

        Parameters:
            lr -- {float} -- learning rate to use when training.

        Functions:
            fit:
                Used to train the model.
                Parameters:
                    image -- {ndarray} -- image used to predict 3D Mesh.
                    X -- {ndarray} -- Data samples - input points coordinates.
                    y -- {ndarray} -- labels for each sample in X
                    epochs -- {int} -- number of epochs to use when training.
    '''

    def __init__(self, output_size, lr=0.01):
        super(Model, self).__init__()
        self.resnet = ResNet(output_size)
        self.meshnet = MeshNet()

    def forward(self, image, X):
        weights = self.resnet(image)
        return self.meshnet(X,weights)


def get_data(path, label_dim, all_indices, grid_dim, samples=2048):
    image_path = path + '\img_choy2016\\'
    files = [image_path + im for im in listdir(image_path) if im[-3:] == 'jpg']
    images = []
    for im in files:
        image = plt.imread(im)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        color = torch.unsqueeze(torch.from_numpy(np.rollaxis(image, 2, 0).copy()).type(torch.float32), 0)
        images.append(color)
        # images.append(torch.unsqueeze(torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).type(torch.float32), 0))


    # grid = trimesh.load(path + '\model.binvox')
    # points = np.load(path + '\pointcloud.npz')

    # create voxel grid
    mesh_grid = trimesh.load(path + '\model.binvox')

    samples_indices = np.random.normal(mesh_grid.sparse_indices, 2).astype(int)
    samples_indices[(samples_indices < 0)] = 0
    samples_indices[(samples_indices > grid_dim - 1)] = grid_dim - 1
    samples_indices = np.unique(samples_indices, axis=0)

    All_indices = all_indices[np.random.randint(0, len(all_indices), int(len(samples_indices) / 10))]

    pos_vg = np.bool_(np.zeros(shape=(grid_dim,grid_dim,grid_dim)))
    pos_vg[mesh_grid.sparse_indices[:, 0], mesh_grid.sparse_indices[:, 1], mesh_grid.sparse_indices[:, 2]] = True


    sample_vg = np.bool_(np.zeros(shape=(grid_dim,grid_dim,grid_dim)))
    sample_vg[samples_indices[:, 0], samples_indices[:, 1], samples_indices[:, 2]] = True
    sample_vg[All_indices[:, 0], All_indices[:, 1], All_indices[:, 2]] = True

    positive_indices = np.argwhere(sample_vg & pos_vg)
    negative_indices = np.argwhere(sample_vg & np.logical_not(pos_vg))

    X = np.concatenate([positive_indices, negative_indices], axis=0)
    y = np.concatenate([np.ones(len(positive_indices)), np.zeros(len(negative_indices))])

    if label_dim == 2:
        y = np.concatenate([1 - y.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    X = torch.Tensor(X)
    X = (X - ((grid_dim-1)/2))/((grid_dim-1)/2)
    y = torch.Tensor(y)

    shuffle_idx = np.arange(len(X))
    np.random.shuffle(shuffle_idx)

    return images, torch.Tensor(X[shuffle_idx][:samples]), torch.Tensor(y[shuffle_idx][:samples])#, color, mesh, trimesh.voxel.VoxelGrid(pos_vg), samples_indices, sample_vg

g_loss = 0.0


def fit(model, optimizer, loss_f, image, X, y, mesh, iterations, batch_size=32, step=True):
    global g_loss
    for i in range(1, iterations + 1):
        yp = model(image, X)
        loss = loss_f(yp, y) / batch_size
        g_loss += float(loss)

        loss.backward()
        if step:
            # print(f"Epoch {i} - [" + "#" * int((i * 20) / iterations) + "-" * (
            #         20 - int((i * 20) / iterations)) + f']  Loss: {g_loss}')
            print(f"Sample {mesh} - [" + "#" * int((i * 20) / iterations) + "-" * (
                    20 - int((i * 20) / iterations)) + f']  Loss: {g_loss}')
            g_loss = 0.0
            optimizer.step()
            optimizer.zero_grad()


folders = r'C:\Users\amita\Desktop\3DReconstruction\shapenet\02691156\\'
mesh_path = listdir(folders)
curr_mesh = mesh_path[2]


curr_folder = mesh_path[2]
files = listdir(folders + curr_folder + '\models\\')

model = Model(2)
loss = nn.BCELoss()

# model = torch.load(r'C:\Users\amita\Desktop\3DReconstruction\model.pt')
model.to(torch.device("cuda:0"))

optimizer = optim.Adam(model.parameters(), lr=0.0000005)

stop = False
# torch.save(model,r'C:\Users\amita\Desktop\3DReconstruction\model.pt')

# angles = np.linspace(0, 360 - 360 / 8, 8).astype(int)
all_indices = np.array([[i, j, k] for i in range(32) for j in range(32) for k in range(32)])
# batch_size = len(angles)
for epoch in range(20):
    for mesh in range(100, 110):
        images, X, y = get_data(folders + mesh_path[mesh], 2, all_indices, 32)
        for i in range(len(images)):
            try:
                fit(model, optimizer, loss, images[i].cuda(), X.cuda(), y.cuda(), f"{mesh}",5)#,len(images),i == (len(images)-1))
            except Exception as e:
                if ("Failed to initialize Pyglet window" in str(e)):
                    stop = True
                    break
                print(e)
        if stop:
            break
        torch.save(model, fr'C:\Users\amita\Desktop\3DReconstruction\model0.pt')
        print("saving_model...")
    if stop:
        break
    print(epoch)


