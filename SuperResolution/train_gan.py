import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from data_utils import DatasetFromFolder
from model_gan import Generator, Discriminator
from psnrmeter import PSNRMeter

import matplotlib.pyplot as plt

import sys

sys.setrecursionlimit(10000000)

train_loss_saver = []
train_psnr_saver = []
val_loss_saver = []
val_psnr_saver = []
best_epoch = []

def processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
    
    optimizer_d.zero_grad()

    fake_images  = generator(data)

    labels_real = torch.ones([target.shape[0], 1, 1, 1]).cuda()
    labels_fake = torch.zeros([fake_images.shape[0], 1, 1, 1]).cuda()

    output_real = discriminator(target)
    output_fake = discriminator(fake_images)
    
    loss_real = criterion_bce(output_real, labels_real)
    loss_fake = criterion_bce(output_fake, labels_fake)
    
    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer_d.step()

    optimizer_g.zero_grad()

    output  = generator(data)
    output_prob = discriminator(output)
    loss_g = criterion_mse(output, target) + 1e-5*torch.sum(-torch.log(output_prob))

    return loss_g, output


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()


def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value())

    train_loss_saver.append(meter_loss.value()[0])
    train_psnr_saver.append(meter_psnr.value())

    reset_meters()

    engine.test(processor, val_loader)
    val_loss_logger.log(state['epoch'], meter_loss.value()[0])
    val_psnr_logger.log(state['epoch'], meter_psnr.value())

    val_loss_saver.append(meter_loss.value()[0])
    val_psnr_saver.append(meter_psnr.value())

    if val_psnr_saver[-1] >= max(val_psnr_saver):
        best_epoch.append(state['epoch'])

    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(generator.state_dict(), 'epochs/gan_epoch_%d_%d.pt' % (UPSCALE_FACTOR, state['epoch']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=200, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    datasetName = 'DIV2K'

    train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, dataset_name=datasetName, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, dataset_name=datasetName, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=5, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=5, batch_size=10, shuffle=False)

    generator = Generator(UPSCALE_FACTOR)
    discriminator = Discriminator()
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss ()
    if torch.cuda.is_available():
        print('Using Device: CUDA')
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_mse = criterion_mse.cuda()
        criterion_bce = criterion_bce.cuda()
    else:
        print('Using Device: CPU')

    print('# parameters:', sum(param.numel() for param in generator.parameters()))

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-5)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Val PSNR'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer_g)
    
    print('The epoch with maximum PSNR is: Epoch '+ str(best_epoch[-1]))
    MODEL_NAME = 'epochs/gan_epoch_'+ str(UPSCALE_FACTOR) + '_' + str(best_epoch[-1]) + '.pt'
    generator.load_state_dict(torch.load(MODEL_NAME))
    resolution_height = int(1920 / UPSCALE_FACTOR / 2)
    resolution_weight = int(3840 / UPSCALE_FACTOR / 2)
    model_path = "pretrained_model/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.onnx.export(generator, torch.randn(1,3,resolution_height,resolution_weight).cuda(), 
                      model_path + "model_" + str(UPSCALE_FACTOR) + '_' + str(NUM_EPOCHS) + '_' + str(resolution_weight) + '_' + str(resolution_height) + ".onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=9,           # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                     )
    

    # plot train loss
    plt.plot(range(len(train_loss_saver)), train_loss_saver)
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Train Loss (SRF = ' + str(UPSCALE_FACTOR) + ')')
    plt.savefig('figures/gan_train_loss_SRF_' + str(UPSCALE_FACTOR) + '.png')
    plt.show()
    

    # plot train psnr
    plt.plot(range(len(train_psnr_saver)), train_psnr_saver)
    plt.xlabel('Epochs')
    plt.ylabel('Train PSNR')
    plt.title('Train PSNR (SRF = ' + str(UPSCALE_FACTOR) + ')')
    plt.savefig('figures/gan_train_psnr_SRF_' + str(UPSCALE_FACTOR) + '.png')
    plt.show()
    

    # plot val loss
    plt.plot(range(len(val_loss_saver)), val_loss_saver)
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss (SRF = ' + str(UPSCALE_FACTOR) + ')')
    plt.savefig('figures/gan_val_loss_SRF_' + str(UPSCALE_FACTOR) + '.png')
    plt.show()
    

    # plot val psnr
    plt.plot(range(len(val_psnr_saver)), val_psnr_saver)
    plt.xlabel('Epochs')
    plt.ylabel('Val PSNR')
    plt.title('Validation PSNR (SRF = ' + str(UPSCALE_FACTOR) + ')')
    plt.savefig('figures/gan_val_psnr_SRF_' + str(UPSCALE_FACTOR) + '.png')
    plt.show()
    

    