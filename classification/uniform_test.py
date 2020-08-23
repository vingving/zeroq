#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--weight-bit', default=4)
    parser.add_argument('--activation-bit', default=4)
    parser.add_argument('--datatype', default='true_data',
                        choices=['distill', 'true_data', 'random_data'])
    parser.add_argument('--distill-size', default=1)
    args = parser.parse_args()
    return args


# def train():
#
# def test():



if __name__ == '__main__':
    args = arg_parse()
    print(args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')

    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/imagenet/',
                              for_inception=args.model.startswith('inception'))

    # Generate distilled data
    if args.datatype == 'distill':
        distill_loader = getDistilData(
            model.cuda(),
            args.distill_size,
            args.dataset,
            batch_size=args.batch_size,
            for_inception=args.model.startswith('inception'))
    elif args.datatype == 'true_data':
        true_loader = getTrainData(args.dataset,
                                  batch_size=args.distill_size,
                                  path='./data/imagenet/',
                                  for_inception=args.model.startswith('inception'))
    elif args.datatype == 'random_data':
        random_loader = getRandomData(dataset=args.dataset,
                                      distill_size=args.distill_size,
                                      batch_size=args.batch_size,
                                      for_inception=args.model.startswith('inception'))
    print('****** Data loaded ******')

    # Quantize single-precision model to 8-bit model
    quantized_model = quantize_model(model, weight_bit=args.weight_bit, activation_bit=args.activation_bit)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # Update activation range according to distilled data
    if args.datatype == 'distill':
        update(quantized_model, distill_loader)
    elif args.datatype == 'true_data':
        update(quantized_model, true_loader, update_single_batch=True)
    elif args.datatype == 'random_data':
        update(quantized_model, random_loader)
    else:
        raise NotImplementedError
    print('****** Zero Shot Quantization Finished ******')

    # for epoch in range(args.epochs):
    #     train_loss, train_acc1 = train(epoch)
    #     test_loss, test_acc1 = test(epoch)
    #
    #     if not args.lr_batch_adj:
    #         lr_scheduler.step()



    # Freeze activation range during test
    freeze_model(quantized_model)
    # quantized_model = nn.DataParallel(quantized_model).cuda()

    # Test the final quantized model
    # test(model, test_loader)
    test(quantized_model, test_loader)


    # Calculate the sensitivity (kl-divergence) between model and quantized model
    # calc_sensitivity(model, quantized_model, visualize=True)


