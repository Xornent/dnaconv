import argparse
from math import exp
import os
import string
import matplotlib
import tqdm
import PIL.Image
import random

import torch
import torch.nn
import torchvision
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as tf
import torch.optim as optim
from torch.optim import sgd
from torch.optim.sgd import SGD;
import torch.distributions as dist
from torch.autograd import Variable;

import numpy

import matplotlib.pyplot as plt
import matplotlib.font_manager

def parseArguments():
    parser = argparse.ArgumentParser( description = """
        trainer.py - trainer program and basic result visualization of dna-circuit convolution classification.
        (partition of in-silico experiments)
    """)

    parser.add_argument ( "-input", dest = "inputSize", type = int, default = 28,
                          help = """original mnist dataset contains 28 x 28 grayscale number images, this can
                          be reshaped to binary images with a specified size""")
    parser.add_argument ( "-debug", dest = "debug", const = True, default = False,
                          help = "enable debug features, including visualization etc.", action = "store_const")

    args = parser.parse_args()
    return args

def main(args):

    nEpoch = 20
    readBatchTrain = 1000
    readBatchTest = 1000
    trainBatch = 20
    testBatch = 1000
    learningRate = 0.0001
    momentum = 0.5
    randomSeed = 42
    torch.manual_seed(randomSeed)
    random.seed(randomSeed)

    # download mnist dataset if the directory 'mnist' does not exist.
    localMnist = os.path.exists('./mnist/')

    trainLoader = None
    testLoader = None
    if not localMnist:
        os.mkdir('./mnist')
        
    trainLoader = DataLoader(
            torchvision.datasets.MNIST('./mnist/', train = True, download = not localMnist,
                               transform = torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor() ])),
            batch_size = readBatchTrain, shuffle = False,)
    testLoader = DataLoader(
            torchvision.datasets.MNIST('./mnist/', train = False, download = not localMnist,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor() ])),
            batch_size=readBatchTest, shuffle = False)
    
    trainImages = {}
    trainTags = []
    testImages = {}
    testTags = []

    # generate or read the generated size of input.
    inputName = 'mnist' + str(args.inputSize) + 'xf'
    localInput = os.path.exists('./' + inputName + '/')

    if not localInput:
        os.mkdir('./' + inputName)
        testTags, testImages = resizeImages(args, testLoader, inputName + '/test')
        trainTags, trainImages = resizeImages(args, trainLoader, inputName + '/train')
    else:
        angles = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
        testTags = numpy.load('./' + inputName + '/test/tags.npy')
        trainTags = numpy.load('./' + inputName + '/train/tags.npy')
        for angle in angles:
            trainImages[angle] = torch.tensor(
                numpy.load('./' + inputName + '/train/' + str(angle) + '.npy')
            )

            testImages[angle] = torch.tensor(
                numpy.load('./' + inputName + '/test/' + str(angle) + '.npy')
            )

            print('Loaded:', str(trainImages[angle].shape), 'from', './' + inputName + '/train/' + str(angle) + '.npy')
    
    networkid = 'final-relu1.pick.24'
    if not os.path.exists('./figures/' + networkid):
        os.mkdir('./figures/' + networkid)

    # only for categories of 1 - 2 are included in the concatencated classfication task.
    concatTrain = []
    concatTag = []
    concatTest = []
    concatTestTag = []

    classes = [2,4]
    
    for id in range(len(trainTags)):
        if trainTags[id] == classes[0] or trainTags[id] == classes[1]:
            if(trainTags[id] == classes[0]):
                concatTag += [1]
            if(trainTags[id] == classes[1]):
                concatTag += [2]
            concatTrain += [trainImages[0][id]]
    
    for id in range(len(testTags)):
        if testTags[id] == classes[0] or testTags[id] == classes[1]:
            concatTest += [testImages[0][id]]

            if(testTags[id] == classes[0]):
                concatTestTag += [1]
            if(testTags[id] == classes[1]):
                concatTestTag += [2]

    pickTrain = [x for x in concatTrain]
    pickTag = [x for x in concatTag]

    constr = weightConstraint()

    net4x44 = Convs4x44()
    
    optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, net4x44.parameters()),
                    lr = learningRate, 
                    momentum = momentum, weight_decay = 0.0001
                )

    ROUNDS = 20

    for epoch in range(ROUNDS):

        trainCount = len(pickTag)
        shuffleId = [j for j in range(len(pickTag))]
        random.shuffle(shuffleId)

        if os.path.exists('./modules/temp' + str(epoch) + '-module-' + networkid + '-b1-lr' + str(learningRate) + '.pt'):
            print('pretrained module exist:', 'temp' + str(epoch) + '-module-' + networkid + '-b1-lr' + str(learningRate) + '.pt')
            net4x44.load_state_dict(torch.load('./modules/temp' + str(epoch) + '-module-' + networkid + '-b1-lr' + str(learningRate) + '.pt'))
            
            test_model(concatTest, concatTestTag, networkid + '-b1-lr' + str(learningRate), epoch)
            if not os.path.exists('./modules/temp' + str(epoch + 1) + '-module-' + networkid + '-b1-lr' + str(learningRate) + '.pt'):
                pass
            
            continue
            
        # learningRate /= 10
        # optimizer = optim.SGD(
        #             filter(lambda p: p.requires_grad, net4x44.parameters()),
        #             lr = learningRate, 
        #             momentum = momentum, weight_decay = 0.0001
        #         )

        losslist = []
        for epoch in range(1):
            rawId = 0
            pbar = tqdm.tqdm(range(trainCount // 1), desc = 'Epoch ' + str(epoch), total = trainCount // 1)
            for it in pbar:
                stackX = []
                ys = []

                for bn in range(1):
                    stackX += [pickTrain[shuffleId[rawId]].unsqueeze(0)]
                    ys += [pickTag[shuffleId[rawId]]]
                    rawId += 1
            
                stackX = torch.stack(stackX, 0)
                with torch.enable_grad():
                    x = Variable(stackX.type(torch.float), requires_grad = True)
                    p = net4x44(x)

                    optimizer.zero_grad()
                    l = torch.zeros(1)
                    for bn in range(1):
                        yp = torch.zeros(2, dtype=torch.float)
                        yp[ys[bn] - 1] = 1
                        yp = Variable(yp, requires_grad = True)

                        l += -( yp[0] * torch.log(p[bn, 0]) + yp[1] * torch.log(p[bn, 1]))
                
                    l /= 1
                    l.backward()
                    pbar.set_description('Epoch ' + str(epoch) + ' Loss ' + str(l.item())) 
                    optimizer.step()
        
                    for mod in net4x44._modules:
                        net4x44._modules[mod].apply(constr)
                        pass

                    losslist += [l.item()]

        torch.save(net4x44.state_dict(), './modules/temp' + str(epoch) + '-module-' + networkid + '-b1-lr' + str(learningRate) + '.pt')

        test_model(concatTest, concatTestTag, networkid + '-b1-lr' + str(learningRate), epoch)
    
    eval_model(concatTest, concatTestTag, networkid + '-b1-lr' + str(learningRate), ROUNDS - 1, networkid)
    pass

@torch.no_grad()
def eval_model(testImages, testTags, modulePath, cpid, networkid):
    net4x44 = Convs4x44()
    statedict = torch.load('./modules/temp' + str(cpid) + '-module-' + modulePath + '.pt')
    # statedict['conn1.bias'] -= 2
    # statedict['conn1.bias'] /= 9.40
    # statedict['conn2.bias'] /= 22.9596
    net4x44.load_state_dict(statedict)
    
    net4x44.eval()

    missing = [[], []]          # no positive category
    coactivate = [[], []]       # more than 1 positive category
    wrong = [[], []]            # one positive category but not truth
    truth = [[], []]            # true prediction
    preds = []

    threshold = 0.7
    for id in tqdm.tqdm(range(len(testTags))):
        corresId = testTags[id] - 1
        x = testImages[id].unsqueeze(0).unsqueeze(0).type(torch.float)
        pred, _ = net4x44(x)

        overThresh = (pred[0] > threshold).type(torch.int)
        sumCat = torch.sum(overThresh)

        if sumCat == 0: missing[corresId] += [id]
        elif sumCat >= 2: coactivate[corresId] += [id]
        elif overThresh[corresId] == 1: truth[corresId] += [id]
        else: wrong[corresId] += [id]

        preds += [pred[0]]
    
    # draw donut chart of performance

    figdonut4x44 = plt.figure(9, dpi = 200, figsize = (3, 3))
    primaryData = [ len(truth[0]), len(truth[1]),
                    len(wrong[0]), len(wrong[1]),
                    len(coactivate[0]), len(coactivate[1]),
                    len(missing[0]), len(missing[1])]
    secondaryData = [ len(truth[0]) + len(truth[1]),
                      len(wrong[0]) + len(wrong[1]),
                      len(coactivate[0]) + len(coactivate[1]),
                      len(missing[0]) + len(missing[1])]
    tertiaryData = [ len(truth[0]) + len(truth[1]),
                     len(wrong[0]) + len(wrong[1]) +
                     len(coactivate[0]) + len(coactivate[1]) + 
                     len(missing[0]) + len(missing[1])]

    primaryColors = ['steelblue', 'darkorchid', 'steelblue', 'darkorchid', 'steelblue', 'darkorchid']
    tertiaryColors = ['green', 'red']
    secondaryColors = ['green', 'red', 'blue', 'gray']

    matplotlib.rcParams['font.family'] = 'Arial'
    plt.pie(primaryData, radius = 1.0, autopct = '%1.1f%%', pctdistance = 1.2, wedgeprops = dict(width = 0.3, edgecolor = 'w'), colors = primaryColors, textprops = dict(fontsize = 6)
            #, explode = (0, 0, 0.2, 0.1, 0, 0, 0, 0))
    )
    plt.pie(secondaryData, radius = 0.7, autopct = None, wedgeprops = dict(width = 0.2, edgecolor = 'w'), colors = secondaryColors)
    plt.pie(tertiaryData, radius = 0.5, autopct = '%1.1f%%', pctdistance = 0.4, wedgeprops = dict(width = 0.1, edgecolor = 'w'), colors = tertiaryColors, textprops = dict(fontsize = 9))
    plt.savefig('./figures/' + networkid + '/evaltest-picker-donut-' + modulePath + '.png')

    f2 = plt.figure(10, dpi = 600, figsize = (3, 3))
    plt.pie(primaryData, radius = 1.0, pctdistance = 1.2, wedgeprops = dict(width = 0.3, edgecolor = 'w'), colors = primaryColors, textprops = dict(fontsize = 6)
            #, explode = (0, 0, 0.2, 0.1, 0, 0, 0, 0))
    )
    plt.pie(secondaryData, radius = 0.7, autopct = None, wedgeprops = dict(width = 0.2, edgecolor = 'w'), colors = secondaryColors)
    plt.pie(tertiaryData, radius = 0.5, pctdistance = 0.4, wedgeprops = dict(width = 0.2, edgecolor = 'w'), colors = tertiaryColors, textprops = dict(fontsize = 9))
    plt.savefig('./figures/' + networkid + '/evaltest-picker-donut-notext-' + modulePath + '.png')
    pass

def resizeImages(args, loader, dirName):
    tags = []
    images = {
            -180: [],
            -150: [],
            -120: [],
            -90: [],
            -60: [],
            -30: [],
            0: [],
            30: [],
            60: [],
            90: [],
            120: [],
            150: []
        }
    
    for batchId, (data, target) in tqdm.tqdm(enumerate(loader), total = len(loader), desc = 'Processing datasets'):
        tags += target
        for rotationAngle in [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]:
            rotatedData = tf.rotate(data, rotationAngle, PIL.Image.BILINEAR,
                    expand = False)
                
            angleImage = []

            for idx in range(len(target)):
                dataImage = rotatedData[idx]
                targetTag = target[idx]
                fromX = 0
                toX = 28
                fromY = 0
                toY = 28
                
                for column in range(28):
                    if torch.sum(dataImage[0, :, column]) <= 0:
                        fromX = column + 1
                    else: break
                
                for column in range(28):
                    if torch.sum(dataImage[0, :, 27 - column]) <= 0:
                        toX = 27 - column
                    else: break

                for row in range(28):
                    if torch.sum(dataImage[0, row, :]) <= 0:
                        fromY = row + 1
                    else: break
                
                for row in range(28):
                    if torch.sum(dataImage[0, 27 - row, :]) <= 0:
                        toY = 27 - row
                    else: break
                    
                trueX = toX - fromX
                trueY = toY - fromY
                paddingX = 0
                paddingY = 0

                if trueX > trueY:
                    paddingY = int((args.inputSize - args.inputSize * trueY / trueX) // 2)
                else:
                    paddingX = int((args.inputSize - args.inputSize * trueX / trueY) // 2)
                    
                trueX = args.inputSize - 2 * paddingX
                trueY = args.inputSize - 2 * paddingY

                tensorImage = torch.zeros([args.inputSize, args.inputSize])
                fixation = tf.resize(dataImage[:, fromY:toY, fromX:toX], [int(trueY), int(trueX)])
                tensorImage[paddingY:paddingY + trueY, paddingX:paddingX + trueX] = fixation[0]
                images[rotationAngle] += [tensorImage]
    
    os.mkdir('./' + dirName + '/')
    for key in images:
        images[key] = torch.stack(images[key], 0)
        images[key] = images[key].type(torch.float)
        numpy.save('./' + dirName + '/' + str(key) + '.npy', images[key].numpy())
        print('Saving data', str(images[key].shape), 'to', './' + dirName + '/' + str(key) + '.npy')
    
    numpy.save('./' + dirName + '/tags.npy', tags)
    return tags, images

@torch.no_grad()
def test_model(testImages, testTags, modulePath, cpid):
    net4x44 = Convs4x44()
    net4x44.load_state_dict(torch.load('./modules/temp' + str(cpid) + '-module-' + modulePath + '.pt'))
    net4x44.eval()

    print('testing ', cpid)
    distances = []
    nums = [0,0]
    truth = [[],[]]

    threshold = 0.6
    for id in tqdm.tqdm(range(len(testTags))):
        corresId = testTags[id] - 1
        x = testImages[id].unsqueeze(0).unsqueeze(0).type(torch.float)
        pred = net4x44(x)
        # pred [1, 2]
        p = pred[0]
        
        yp = torch.zeros(2, dtype=torch.float)
        yp[corresId] = 1
        # pred[0, corresId] = 1 - pred[0, corresId]
        l = - ( yp[0] * torch.log(p[0]) 
              + yp[1] * torch.log(p[1]))
        #distance = exp(pred[0,0]) + exp(pred[0,1]) + exp(pred[0,2]) + exp(pred[0,3]) - 4
        distances += [l.item()]

    for id in range(len(testTags)):
        corresId = testTags[id] - 1
        x = testImages[id].unsqueeze(0).unsqueeze(0).type(torch.float)
        pred = net4x44(x)

        overThresh = (pred[0] > threshold).type(torch.int)
        sumCat = torch.sum(overThresh)

        nums[corresId] += 1
        
        if sumCat == 0: pass
        elif sumCat >= 2: pass
        elif overThresh[corresId] == 1: 
            truth[corresId] += [id]
        else: pass
    
    print ('truth in test fitting threshold: (',len(truth[0]),len(truth[1]),')')
    print('test classes:', nums[0], nums[1])

torch.enable_grad()
class Convs4x44 (torch.nn.Module):

    def __init__(self):
        super(Convs4x44, self).__init__()

        self.conv1 = torch.nn.Conv2d( in_channels = 1, out_channels = 1, 
                                      kernel_size = (2,2), stride = (2,2), bias = False )

        self.conv2 = torch.nn.Conv2d( in_channels = 1, out_channels = 1, 
                                     kernel_size = (2,2), stride = (2,2), bias = False )

        self.conn1 = torch.nn.Linear( in_features = 4, out_features = 3, bias = True)
        self.conn2 = torch.nn.Linear( in_features = 3, out_features = 2, bias = True)

        torch.nn.init.normal_(self.conv1.weight.data, 0.5, 0.15)
        torch.nn.init.normal_(self.conv2.weight.data, 0.5, 0.15)
        torch.nn.init.normal_(self.conn1.weight.data, 0.5, 0.15)
        torch.nn.init.normal_(self.conn2.weight.data, 0.5, 0.15)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = (x1 - 0.2).relu()
        x1 = x1 # / 3.8

        x2 = self.conv2(x1)
        
        batch, _, _, _ = x2.shape
        classification = []
        rawclass = []
        for batchId in range(batch):

            inp_fullconn = x2[batchId].reshape(4).unsqueeze(0)
            inp_fullconn = (inp_fullconn - 2.0).relu()
            inp_fullconn = inp_fullconn # / 2.4737
            
            inp_interm = self.conn1(inp_fullconn)
            inp_interm = (inp_interm).relu()
            inp_interm = inp_interm # / 2.4425

            outp = self.conn2(inp_interm)

            outp = outp # * 22.9596

            classify = torch.zeros(2)
            classify[0] = outp[0, 0]
            classify[1] = outp[0, 1]

            expsum = torch.sum( torch.exp(classify) )
            rawclass += [outp[0]]
            classification += [torch.exp(classify)/expsum]
        
        return torch.stack(classification, 0), torch.stack(rawclass, 0)
        
class weightConstraint():
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0, 1)
            module.weight.data = w

args = parseArguments()
main(args)