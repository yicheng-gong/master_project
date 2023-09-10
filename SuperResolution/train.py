import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from data_utils import DatasetFromFolder
from model import Net
from psnrmeter import PSNRMeter

import matplotlib.pyplot as plt

train_loss_saver = []
train_psnr_saver = []
val_loss_saver = []
val_psnr_saver = []

def processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    loss = criterion(output, target)

    return loss, output


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
    scheduler.step()
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

    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), 'epochs/epoch_%d_%d.pt' % (UPSCALE_FACTOR, state['epoch']))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    datasetName = 'DIV2K'

    train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, dataset_name=datasetName, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, dataset_name=datasetName, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=10, shuffle=False)

    model = Net(UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        print('Using Device: CUDA')
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print('Using Device: CPU')

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

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

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
    
    resolution_height = int(1920 / UPSCALE_FACTOR / 2)
    resolution_weight = int(3840 / UPSCALE_FACTOR / 2)
    model_path = "models_onnx/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.onnx.export(model, torch.randn(1,3,resolution_height,resolution_weight).cuda(), 
                      model_path + "model_" + str(UPSCALE_FACTOR) + '_' + str(NUM_EPOCHS) + '_' + str(resolution_weight) + '_' + str(resolution_height) + ".onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=9,           # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                     )
    
    # plot train loss
    plt.plot(range(len(train_loss_saver)), train_loss_saver)
    plt.xlabel('Iterations')
    plt.ylabel('Train Loss')
    plt.title('Train Loss')
    plt.savefig('figures/ESPCN_train_loss_SRF_' + str(UPSCALE_FACTOR) + '.pdf')
    plt.show()
    

    # plot train psnr
    plt.plot(range(len(train_psnr_saver)), train_psnr_saver)
    plt.xlabel('Iterations')
    plt.ylabel('Train PSNR')
    plt.title('Train PSNR')
    plt.savefig('figures/ESPCN_train_psnr_SRF_' + str(UPSCALE_FACTOR) + '.pdf')
    plt.show()
    

    # plot val loss
    plt.plot(range(len(val_loss_saver)), val_loss_saver)
    plt.xlabel('Iterations')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss')
    plt.savefig('figures/ESPCN_val_loss_SRF_' + str(UPSCALE_FACTOR) + '.pdf')
    plt.show()
    

    # plot val psnr
    plt.plot(range(len(val_psnr_saver)), val_psnr_saver)
    plt.xlabel('Iterations')
    plt.ylabel('Val PSNR')
    plt.title('Validation PSNR')
    plt.savefig('figures/ESPCN_val_psnr_SRF_' + str(UPSCALE_FACTOR) + '.pdf')
    plt.show()
    

    