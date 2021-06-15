import os
import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from my_models import ResNet18, ResNet50,ResNet101
from dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from evaluator import getAUC, getACC, save_results
from info import INFO

device = torch.device("cuda:1")
flag_to_class = {
    "pathmnist": PathMNIST,
    "chestmnist": ChestMNIST,
    "dermamnist": DermaMNIST,
    "octmnist": OCTMNIST,
    "pneumoniamnist": PneumoniaMNIST,
    "retinamnist": RetinaMNIST,
    "breastmnist": BreastMNIST,
    "organmnist_axial": OrganMNISTAxial,
    "organmnist_coronal": OrganMNISTCoronal,
    "organmnist_sagittal": OrganMNISTSagittal,
}


def main(flag, input_root, output_root, end_epoch, model_name):
    download = False
    DataClass = flag_to_class[flag]

    info = INFO[flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    num_workers = 4

    start_epoch = 0
    lr = 0.001
    batch_size = 128
    val_auc_list = []
    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_transform = transforms.Compose([
                                        transforms.ColorJitter(brightness = 0.5, contrast = 0.5,satuation = 0.5),
                                        transforms.RandomVerticalFlip(p = 0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[.5], std=[.5])])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=input_root,split='train',transform=train_transform,download=download)
    train_loader = data.DataLoader(dataset=train_dataset,batch_size=batch_size,num_workers = num_workers,shuffle=True)

    val_dataset = DataClass(root=input_root,split='val',transform=transform,download=download)
    val_loader = data.DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers = num_workers,shuffle=True)

    test_dataset = DataClass(root=input_root,split='test',transform=transform,download=download)
    test_loader = data.DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers = num_workers,shuffle=True)
    
    if model_name == 18:
        model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
    elif model_name == 50:
        model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
    else:
        model = ResNet101(in_channels=n_channels, num_classes=n_classes).to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, dir_path, epoch)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(model_path)['net'])
    test(model,
         'train',
         train_loader,
         device,
         flag,
         task,
         output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)
    test(model,
         'test',
         test_loader,
         device,
         flag,
         task,
         output_root=output_root)


def train(model, optimizer, criterion, train_loader, device, task):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, task, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task, output_root=None):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--model_name',
                        default = 18,
                        help = 'resnet18 or resnet 50',
                        type = int)
    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    model_name = args.model_name
    main(data_name,input_root,output_root,end_epoch, model_name)
