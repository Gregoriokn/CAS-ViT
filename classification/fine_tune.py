import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
import json
import torch
import numpy as np
import os
import shutil
from types import SimpleNamespace
from pathlib import Path
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from torch import optim
from model import *
from model import RCvitAdapter
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from manager_dataset import CustomDataset, get_tt_split
import utils as utils

from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset


def get_args_parser():
    parser = argparse.ArgumentParser('CAS-ViT training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=2, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='rcvit_xs', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=6e-3, metavar='LR',
                        help='learning rate (default: 6e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--warmup_start_lr', type=float, default=0, metavar='LR',
                        help='Starting LR for warmup (default 0)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/imagenet_full', type=str,
                        help='dataset path (path to full imagenet)')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=43, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--finetune', default='',
                        help='finetune the model')
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--adapters',action="store_true",
                        help='choose the use of adapters')
    #parser.add_argument('--nb_classes', default=1000, type=int,
    #                    help='number of the classification types')

    return parser.parse_args()

def training_routine(model, patience_time, dl_train, dl_valid,lr, device, nome_do_modelo):
    loss_train = []
    loss_eval  = []
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    epochs = 100

    stop = False
    epoch = 0
    lowest_loss_eval = 10000
    last_best_result = 0
    while (not stop):
        model.train()
        lloss = []
        for x,y in dl_train:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            closs = criterion(pred,y)
            closs.backward()
            opt.step()
            opt.zero_grad()
            lloss.append(closs.item())
        loss_train.append(np.mean(lloss))
        lloss = []
        model.eval()
        lres = []
        ytrue = []
        with torch.no_grad():
            for data,y in dl_valid:
                data = data.to(device)

                pred = model(data)
                closs = criterion(pred.cpu(),y)
                lloss.append(closs.item())
                res  = pred.argmax(dim=1).cpu().tolist()
                lres += res
                ytrue += y
        avg_loss_eval = np.mean(lloss)
        loss_eval.append(avg_loss_eval)
        #wandb.log({"loss_eval": avg_loss_eval,"loss_train":loss_train[-1]})
        if avg_loss_eval < lowest_loss_eval:
            lowest_loss_eval = avg_loss_eval
            last_best_result = 0
            print("Best model found! saving...")
            actual_state = {'optim':opt.state_dict(),'nome_do_modelo':nome_do_modelo,'model':model.state_dict(),'epoch':epoch,'loss_train':loss_train,'loss_eval':loss_eval}
            torch.save(actual_state, f'{nome_do_modelo}_best_model.pth')
        last_best_result += 1
        if last_best_result > patience_time:
            stop = True
        print("epoch %d loss_train %4.3f loss_eval %4.3f last_best %d"%(epoch,loss_train[-1],loss_eval[-1],last_best_result))
        epoch += 1

def fine_tune():
    args = get_args_parser()
    transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter if args.color_jitter > 0 else None,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    #dataset = CustomDataset(args.data_path, transform=transform)
    #dl_train, dl_val, dl_test = get_tt_split(dataset, args.batch_size)

    # Verifica se as subpastas 'train' e 'test' existem
    train_dir = os.path.join(args.data_path, 'train')
    test_dir = os.path.join(args.data_path, 'test')
    Classes = 0

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        # Carrega os datasets a partir das subpastas 'train' e 'test'
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
        print("Numero de classes:"+ str(len(train_dataset.classes)))
        Classes = len(train_dataset.classes)
        
        # Concatena os datasets de treino e teste
        #full_dataset = ConcatDataset([train_dataset, test_dataset])
         # Divide o dataset de treino em treino e validação
        train_size = int(0.7 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
    else:
        # Carrega o dataset a partir da pasta principal
        full_dataset = ImageFolder(args.data_path, transform=transform)
        #   Define os tamanhos para treino, validação e teste
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
    
        # Divide o dataset completo
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        print("Numero de classes:"+ str(len(full_dataset.classes)))
        Classes = len(full_dataset.classes)
    
    # Divide o dataset completo
    #train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(len(train_dataset)+len(test_dataset))
    print(train_size)
    print(val_size)
    print(len(test_dataset))
    
    # DataLoaders
    dl_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    dl_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = (
    torch.device('mps') if torch.backends.mps.is_available() else
    torch.device('cuda:0') if torch.cuda.is_available() else
    torch.device('cpu')
    )

    print(f"Running on {device}")

    if(args.adapters == True):
        tuning_config = SimpleNamespace(
        _device = device,
        ffn_num = 96,
        ffn_adapt = "True",
        ffn_option = "parallel",
        ffn_adapter_init_option = 'lora',
        ffn_adapter_scalar = 1.0,
        ffn_adapter_layernorm_option = "in"
        )
        nome_do_modelo = "RCvitAdapter"
        print("RCvitAdapter")
        model = RCvitAdapter(
            layers = [2, 2, 4, 2],
            embed_dims=[48, 56, 112, 220],
            model = args.model,
            finetune = args.finetune,
            pretrained=False,
            num_classes=Classes,
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
            head_init_scale=1.0,
            input_size=args.input_size,
            classifier_dropout=0.0,
            distillation=False,
            tuning_config = tuning_config,
            training = True,
            device = device
            
        )
        
    else:
        print("RCvit")
        nome_do_modelo = "RCVit"
        model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        input_res=384,
        classifier_dropout=0.0,
        distillation=False
    )
        checkpoint = torch.load(args.finetune, map_location="cpu")
        state_dict = checkpoint["model"]
        utils.load_state_dict(model, state_dict)
        model.head = nn.Linear(in_features=220, out_features=Classes, bias=True)
    
    model.to(device)

    training_routine(model, 15, dl_train, dl_valid=dl_val, lr=args.lr, device=device, nome_do_modelo=nome_do_modelo)
    
    model.eval()
    lres = []
    ytrue = []
    with torch.no_grad():
        for data,target in dl_test:
            data = data.to(device)
            pred = model(data)
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += target
    #target_names = ['Cachorro', 'Calopsita', 'Gato'] 
    print(classification_report(ytrue,lres))
fine_tune()
