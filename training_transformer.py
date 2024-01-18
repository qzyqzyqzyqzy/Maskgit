import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        train_dataset1,train_dataset2 = load_data(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset1))) as pbar:
                for i, (img1,img2) in zip(pbar, zip(train_dataset1,train_dataset2)):
                    imglist=[img1,img2]
                    for img in imglist:
                        imgs=img[0]
                        #for j in range(4):#这里是把一个batch分为几个小batch多次更新权重，只要把batch调大，可以实现任意的连续训练
                        #如果想在某个size的图片上面连续训练一定次数，只要把用for循环，imgs[j:j+?,:,:,:]截取出来 
                        self.optim.zero_grad()
                        imgs = imgs.to(device=args.device)
                        logits, targets = self.model(imgs)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        loss.backward()
                        self.optim.step()
                        pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                        pbar.update(0)
            log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(sampled_imgs, os.path.join("results", f"transformer_{epoch}.jpg"), nrow=4)
            plot_images(log)
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
    
    
# basic hyperparameters
    parser.add_argument('--resolution', type=int, default=64, help='resolution')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=100, help='when sampling,choose from the top 100')
    parser.add_argument('--latent-dim', type=int, default=512, help='Latent dimension n_z.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--batch-size', type=int, default=2, help='Input batch size for training.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/vqgan_epoch_0.pt', help='Path to checkpoint.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    args = parser.parse_args()


    train_transformer = TrainTransformer(args)


