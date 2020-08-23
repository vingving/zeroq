import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def plot_sensitivity(kl_loss):
    plt.plot(kl_loss,'-o')
    #plt.plot(history.history['val_loss'],'-o')
    plt.title('Sensitivity')
    plt.ylabel('sensitivity')
    plt.xlabel('layers')
    #plt.legend(['Train', 'Test'], loc=0)


def calc_sensitivity(model, quantized_model, visualize=False):

    # Calculate the sensitivity (kl-divergence) between model and quantized model
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

    if visualize:
        plot_sensitivity(kl_loss)
        plt.show()
