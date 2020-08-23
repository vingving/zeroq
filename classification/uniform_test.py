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
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *
from torch.autograd import Variable
import matplotlib.pyplot as plt


def plot_sensitivity(kl_loss):
    plt.plot(kl_loss,'-o')
    #plt.plot(history.history['val_loss'],'-o')
    plt.title('Sensitivity')
    plt.ylabel('sensitivity')
    plt.xlabel('layers')
    #plt.legend(['Train', 'Test'], loc=0)


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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
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
    dataloader = getDistilData(
        model.cuda(),
        args.dataset,
        batch_size=args.batch_size,
        for_inception=args.model.startswith('inception'))
    print('****** Data loaded ******')

    # Quantize single-precision model to 8-bit model
    quantized_model = quantize_model(model)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # Update activation range according to distilled data
    update(quantized_model, dataloader)
    print('****** Zero Shot Quantization Finished ******')

    # Freeze activation range during test
    freeze_model(quantized_model)
    # quantized_model = nn.DataParallel(quantized_model).cuda()

    # Test the final quantized model
    test(model, test_loader)
    test(quantized_model, test_loader)
    #quantized_model.save('quantized_resnet18.pth')


    # Calculated the sensitivity (kl-divergence) between model and quantized model
    model_params = ['features.init_block.conv.conv.weight', 'features.stage1.unit1.body.conv1.conv.weight',
                    'features.stage1.unit1.body.conv2.conv.weight', 'features.stage1.unit2.body.conv1.conv.weight',
                    'features.stage1.unit2.body.conv2.conv.weight', 'features.stage2.unit1.body.conv1.conv.weight',
                    'features.stage2.unit1.body.conv2.conv.weight', 'features.stage2.unit2.body.conv1.conv.weight',
                    'features.stage2.unit2.body.conv2.conv.weight', 'features.stage3.unit1.body.conv1.conv.weight',
                    'features.stage3.unit1.body.conv2.conv.weight', 'features.stage3.unit2.body.conv1.conv.weight',
                    'features.stage3.unit2.body.conv2.conv.weight', 'features.stage4.unit1.body.conv1.conv.weight',
                    'features.stage4.unit1.body.conv2.conv.weight', 'features.stage4.unit2.body.conv1.conv.weight',
                    'features.stage4.unit2.body.conv2.conv.weight', 'output.weight']

    quantized_model_params = ['features.0.conv.conv.weight', 'features.1.0.body.conv1.conv.weight',
                              'features.1.0.body.conv2.conv.weight', 'features.1.1.body.conv1.conv.weight',
                              'features.1.1.body.conv2.conv.weight', 'features.2.0.body.conv1.conv.weight',
                              'features.2.0.body.conv2.conv.weight', 'features.2.1.body.conv1.conv.weight',
                              'features.2.1.body.conv2.conv.weight', 'features.3.0.body.conv1.conv.weight',
                              'features.3.0.body.conv2.conv.weight', 'features.3.1.body.conv1.conv.weight',
                              'features.3.1.body.conv2.conv.weight', 'features.4.0.body.conv1.conv.weight',
                              'features.4.0.body.conv2.conv.weight', 'features.4.1.body.conv1.conv.weight',
                              'features.4.1.body.conv2.conv.weight', 'output.weight']
    kl_loss = []
    for idx in range(len(model_params)) :
        model_val = Variable(model.state_dict()[model_params[idx]])
        quantized_model_val = Variable(quantized_model.state_dict()[quantized_model_params[idx]])
        #print(model_val.shape)
        #print(quantized_model_val.shape)
        #kl_loss.append(
        #    torch.nn.functional.kl_div(model_val, quantized_model_val))
        kl = []
        for b in range(model_val.size(0)):
            src = model_val[b]
            tgt = quantized_model_val[b]
            src -= src.min()
            src /= src.max()
            tgt -= tgt.min()
            tgt /= tgt.max()
            kl.append(torch.nn.KLDivLoss(size_average=True)(F.log_softmax(src, -1), tgt).cpu().numpy())
        kl_loss.append(np.mean(kl))
    plot_sensitivity(kl_loss)
    plt.show()
    #print(kl_loss)

