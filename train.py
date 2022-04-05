import argparse
from pathlib import Path

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import function
import copy

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True

def prune_by_percentile(model, percent, reinit=False, **kwargs):
    global step
    global mask
    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)

def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0


def original_initialization(model, mask_temp, initial_state_dict):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=1)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
parser.add_argument("--prune_iterations", default=2, type=int, help="Pruning iterations count")
args = parser.parse_args()

device = torch.device('cuda')
reinit = True if args.prune_type=="reinit" else False
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))
mask = []

decoder = net.decoder
vgg = net.vgg

# Weight Initialization
decoder.apply(weight_init)
make_mask(decoder)

# encoder weight init
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

# Copying and Saving Initial State
initial_state_dict = copy.deepcopy(decoder.state_dict())
function.checkdir(f"{os.getcwd()}/saves/")
torch.save(decoder, f"{os.getcwd()}/saves/initial_state_dict_{args.prune_type}.pth.tar")

network = net.Net(vgg, decoder)
network.train()
network = network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

# Layer Looper
for name, param in decoder.named_parameters():
    print(name, param.size())

# Pruning
# NOTE First Pruning Iteration is of No Compression
ITERATION = args.prune_iterations
comp = np.zeros(ITERATION,float)
step = 0
all_loss = np.zeros(args.max_iter,float)

for _ite in range(1, ITERATION):
    if not _ite == 0:
        prune_by_percentile(decoder, args.prune_percent, reinit=reinit)
        if reinit:
            decoder.apply(weight_init)
            step = 0
            for name, param in decoder.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.numpy() * mask[step]).to(weight_dev)
                    step = step + 1
            step = 0
        else:
            original_initialization(decoder, mask, initial_state_dict)
            decoder = decoder.cuda()
    print(f"\n--- Pruning Level [{_ite}/{ITERATION}]: ---")

    # Print the table of Nonzeros in each layer
    comp1 = function.print_nonzeros(decoder)
    comp[_ite] = comp1
    pbar = tqdm(range(args.max_iter))

    for i in pbar:
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss_c, loss_s = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s
        all_loss[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cuda'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))

        # Frequency for Printing Loss
        pbar.set_description(
                f'Train Epoch: {i}/{args.max_iter} Loss: {loss:.6f}')

        # Dump Plot values
        function.checkdir(f"{os.getcwd()}/dumps/")
        all_loss.dump(
            f"{os.getcwd()}/dumps/all_loss_{comp1}.dat")

        # Making variables into 0
        all_loss = np.zeros(args.max_iter, float)
    # Dumping Values for Plotting
    function.checkdir(f"{os.getcwd()}/dumps/")
    comp.dump(f"{os.getcwd()}/dumps/compression.dat")

writer.close()
