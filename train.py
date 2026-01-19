import argparse
import glob
import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from PIL import Image
import time
import os

from model.net import Net


class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, train_glob, transform):
        super(DIV2KDataset, self).__init__()
        self.transform = transform
        self.images = list(sorted(glob.glob(train_glob)))

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # 这里加个异常处理，防止某张图坏了导致整个训练中断
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"[Warning] Error loading image {img_path}: {e}")
            # 如果出错，随机返回一张全黑图防止崩坏（或者更好的处理是跳过，但Dataset里跳过比较麻烦）
            return torch.zeros((3, 256, 256))

    def __len__(self):
        return len(self.images)


class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        img /= 127.5
        img -= 1.0
        return img.transpose((2, 0, 1))


def train(train_data_path, lmbda, lr, batch_size, checkpoint_dir, weight_path, is_high, post_processing):
    # === 核心修改在这里 ===
    # 加入 Resize(256)，确保图片最小边至少是 256，解决 251<256 的报错
    full_dataset = DIV2KDataset(train_data_path, transform=tv.transforms.Compose([
        tv.transforms.Resize((256, 256)),  # [新增] 强制调整大小，防止图片过小报错
        tv.transforms.RandomCrop(256),  # [原有] 随机裁剪
        Preprocess()
    ]))

    # 2. 计算划分长度 (80% 训练, 20% 验证)
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"[INFO] Dataset split: {train_size} training images, {val_size} validation images.")

    # 3. 创建 DataLoader
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    net = Net((batch_size, 256, 256, 3), (1, 256, 256, 3), is_high, post_processing).cuda()

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    if weight_path != "":
        print(f"[INFO] Loading weights from {weight_path}")
        net.load_state_dict(torch.load(weight_path), strict=True)
        try:
            ckpt_name = os.path.basename(weight_path)
            start_epoch = int(os.path.splitext(ckpt_name)[0]) + 1
            print(f"[INFO] Resuming from epoch {start_epoch}")
        except:
            start_epoch = 0
    else:
        net.apply(weight_init)
        start_epoch = 0

    if post_processing:
        opt = optim.Adam(net.post_processing_params(), lr=lr)
        sch = optim.lr_scheduler.MultiStepLR(opt, [1200, 1350], 0.5)
        train_epoch = 1500
    else:
        opt = optim.Adam(net.base_params(), lr=lr)
        sch = optim.lr_scheduler.MultiStepLR(opt, [4000, 4500, 4750], 0.5)
        train_epoch = 5000

    net = nn.DataParallel(net)

    for epoch in range(start_epoch, train_epoch):
        net.train()

        list_train_loss = 0.
        list_train_bpp = 0.
        list_train_mse = 0.
        cnt = 0

        for i, data in enumerate(training_loader, 0):
            x = data.cuda()
            opt.zero_grad()
            train_bpp, train_mse = net(x, 'train')

            train_loss = lmbda * train_mse + train_bpp
            train_loss = train_loss.mean()
            train_bpp = train_bpp.mean()
            train_mse = train_mse.mean()

            if np.isnan(train_loss.item()):
                raise Exception('NaN in loss')

            list_train_loss += train_loss.item()
            list_train_bpp += train_bpp.item()
            list_train_mse += train_mse.item()

            train_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            opt.step()
            cnt += 1

        avg_train_loss = list_train_loss / cnt
        avg_train_bpp = list_train_bpp / cnt
        avg_train_mse = list_train_mse / cnt

        # === 验证阶段 ===
        net.eval()
        list_val_loss = 0.
        list_val_bpp = 0.
        list_val_mse = 0.
        val_cnt = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                x = data.cuda()
                val_bpp, val_mse = net(x, 'train')
                val_loss = lmbda * val_mse + val_bpp

                list_val_loss += val_loss.mean().item()
                list_val_bpp += val_bpp.mean().item()
                list_val_mse += val_mse.mean().item()
                val_cnt += 1

        avg_val_loss = list_val_loss / val_cnt if val_cnt > 0 else 0
        avg_val_bpp = list_val_bpp / val_cnt if val_cnt > 0 else 0
        avg_val_mse = list_val_mse / val_cnt if val_cnt > 0 else 0

        print('[Epoch %04d] Train Loss: %.4f (Bpp: %.4f MSE: %.4f) | Val Loss: %.4f (Bpp: %.4f MSE: %.4f)' % (
            epoch,
            avg_train_loss, avg_train_bpp, avg_train_mse,
            avg_val_loss, avg_val_bpp, avg_val_mse
        )
              )

        sch.step()

        if (epoch % 100 == 99):
            print('[INFO] Saving checkpoint...')
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(net.module.state_dict(), '%s/%04d.ckpt' % (checkpoint_dir, epoch))

        with open(os.path.join(checkpoint_dir, 'train_log.txt'), 'a') as fd:
            fd.write('[Epoch %04d] Train Loss: %.4f Val Loss: %.4f \n' % (epoch, avg_train_loss, avg_val_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_data_path", default="../DIV2K", help="Directory of Testset Images")
    parser.add_argument("--weight_path", default="", help="Path of Pretrained Checkpoint")
    parser.add_argument("--checkpoint_dir", default="./ckpt", help="Directory of Saved Checkpoints")
    parser.add_argument("--high", action="store_true", help="Using High Bitrate Model")
    parser.add_argument("--post_processing", action="store_true", help="Using Post Processing")
    parser.add_argument("--lambda", type=float, default=0.01, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")

    args = parser.parse_args()

    train(args.train_data_path, args.lmbda, args.lr, args.batch_size, args.checkpoint_dir, args.weight_path, args.high,
          args.post_processing)