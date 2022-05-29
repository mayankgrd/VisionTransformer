from lib2to3.pgen2 import pgen
import os
from pickletools import optimize
import torch
from torch import nn, normal
import torch.nn.functional as F
from torchvision.datasets import CIFAR10 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torchmetrics
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from argparse import ArgumentParser

from hello_vit import VisionTransformer 


class lighteningModel(pl.LightningModule):
    def __init__(self, model, learning_rate = 1e-3):
        super().__init__()
        
        self.model = model 
        self.learning_rate = learning_rate


        self.loss = nn.CrossEntropyLoss()
        self.metric = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx): 
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss 

    def validation_step(self, batch, batch_idx):
        X, y = batch 
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        # compute accuracy here 
        acc = self.metric(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) 
        self.log('val_acc', acc, on_step=True, prog_bar = True,logger=True)
    def validation_epoch_end(self, validation_step_output) -> None:
        super().validation_epoch_end(validation_step_output) 

        acc_epoch = self.metric.compute()
        self.metric.reset()
        self.log('val_acc_epoch', acc_epoch, on_step =False, on_epoch=True, prog_bar=True, logger=True)





    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer 




def get_datasets(batch_size =16, num_workers = 2, normalize=False): 

    if normalize:
        normalize_tr = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        normalize_tr = transforms.Normalize((0, 0, 0), (1,1,1))
    

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_tr
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_tr
    ])

    trainset =datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes 



if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument_group('trainer')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    
    trainloader, testloader, classes  = get_datasets(batch_size=args.batch_size, num_workers=args.num_workers, normalize=True)
    ViT = VisionTransformer(img_size=32, patch_size=4, embed_dim=48) 
    ViT_pl = lighteningModel(ViT, learning_rate=args.learning_rate)


    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(ViT_pl,train_dataloaders=trainloader, val_dataloaders=testloader)


    # save the model that is trained 
    trainer.save_checkpoint("final.ckpt")