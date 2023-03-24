# -*- coding = utf-8 -*-  
# @Time: 2023/3/24 04:15 
# @Author: Dylan 
# @File: train.py 
# @software: PyCharm
import os.path
import math
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from utils import *
from my_dataset import *
from model import *

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(args)

    print('Start Tensorboard with "tensorboard -- login-runs", view at http://localhost:6006 ')
    tb_writer = SummaryWriter()

    if os.path.exists('./weights') is False:
        os.makedirs("./weights")

    train_image_path, train_images_labels, val_images_path, val_images_labels = read_split_data(args.data_path)

    # 数据处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        "val": transforms.Compose([transforms.RandomResizedCrop((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # 实例化数据集
    train_dataset = MyDataset(images_path=train_image_path,
                              images_class=train_images_labels,
                              transform=data_transform['train'])

    val_dataset = MyDataset(images_path = val_images_path,
                            images_class=val_images_labels,
                            transform=data_transform['val'])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('using {} dataloader workers every process'.format(nw))

    # 加载数据集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=nw,
                                          collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model_name = 'vgg16'
    model = vgg(model_name=model_name, num_class=args.num_class).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file:'{}' not exist".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # freeze all the layers except the last conv and fc layers
            if ("features.top" not in name) and ('classifier' not in name):
                para.requires_grad(False)

            else:
                print('training{}'.format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params=pg, lr=args.lr)

    # scheduler
    # scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device,
                       epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # dataset path

    parser.add_argument('--data-path', type=str,
                        default='/Users/chengshengdeng/Documents/MachineLearning/DeepLearning/Pytorchlearning/'
                                'deep-learning-for-image-processing-master/data_set/flower_data/flower_photos')

    # download model weights
    parser.add_argument('--weights', type=str, default='./vgg16.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 1, or cpu')

    opt = parser.parse_args()
    main(opt)













