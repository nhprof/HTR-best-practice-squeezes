import argparse
import logging
import datetime as dt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from config_squeeze import *


from utils.iam_dataset import IAMDataset
from utils.rimes_dataset import RimesDataset
from utils.squeeze_dataset import SqueezeDataset
from utils.dbl_squeeze_dataset import DoubleSqueezeDataset
from utils.squeeze_detected_dataset import SqueezeDetectedDataset
from utils.dbl_squeeze_detected_dataset import DoubleSqueezeDetectedDataset
from models_squeeze import HTRNet

import editdistance

# (don't) Create tensorflow output file
start = dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
#writer = SummaryWriter(log_dir="logs/model_" + start)


# Create a logger
logger = logging.getLogger('HTR-Experiment::test')
logger.setLevel(logging.DEBUG)

# Create a formatter to define the log format
formatter = logging.Formatter('[%(asctime)s, %(levelname)s, %(name)s] %(message)s')

# Create a file handler to write logs to a file
file_handler = logging.FileHandler('logs/test_htr_'+start)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create a stream handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
#logger = logging.getLogger('HTR-Experiment::test')
#logger.setLevel(logging.INFO)
#logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S',
#                    level=logging.INFO,
#                    filename='logs/htr_'+dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), encoding='utf-8')

logger.info('--- Running HTR Testing ---')
# argument parsing
parser = argparse.ArgumentParser()
# - test arguments
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--dataset', action='store', type=str, default='SqueezeDetIAS')
parser.add_argument('--weight_path', action='store', type=str, default='trained_models/SqueezeIAS_E640_2024-08-29_09_57_26.pt')
parser.add_argument('--head_layers', action='store', type=int, default=3)
parser.add_argument('--head_type', action='store', type=str, default='cnn')

args = parser.parse_args()

gpu_id = args.gpu_id
logger.info('###########################################')

# prepare datset loader
(state,classes) = torch.load(args.weight_path)

squeeze_transforms = []

logger.info('Loading dataset: '+args.dataset)

if args.dataset == 'SqueezeDetIAS':
    trainDataset = SqueezeDataset
    myDataset = SqueezeDetectedDataset
    dataset_folder = 'C:/Research/Squeezes'
elif args.dataset == 'DblSqueezeDetIAS':
    trainDataset = DoubleSqueezeDataset
    myDataset = DoubleSqueezeDetectedDataset
    dataset_folder = 'C:/Research/Squeezes'
else:
    raise NotImplementedError

train_set = None
val_set = None
test_set = None
if args.dataset == 'SqueezeDetIAS' or  args.dataset == 'DblSqueezeDetIAS':
    #train_set = myDataset(dataset_folder, 'train', level, fixed_size=fixed_size, character_classes = classes, transforms=None, line_dir='{}/Extended Detected Lines')
    #print('# training lines ' + str(train_set.__len__()))
    val_set = myDataset(dataset_folder, 'val', level, fixed_size=fixed_size, character_classes = classes, transforms=None, line_dir='{}/Extended Detected Lines')
    print('# validation lines ' + str(val_set.__len__()))
    test_set = myDataset(dataset_folder, 'test', level, fixed_size=fixed_size, character_classes = classes, transforms=None, line_dir='{}/Extended Detected Lines')
    print('# testing lines ' + str(test_set.__len__()))

    
# if not ' ' in classes:
#     classes.insert(0,' ')
# if '_' in classes:
#     classes.remove('_')
    
# classes = '_'+''.join(classes)

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

# augmentation using data sampler
if train_set is not None:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
if val_set is not None:
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# load CNN
logger.info('Preparing Net...')

if args.head_layers > 0:
    head_cfg = (head_cfg[0], args.head_layers)

head_type = args.head_type

net = HTRNet(cnn_cfg, head_cfg, len(classes), head=head_type, flattening=flattening, stn=stn, inlayer=test_set.nlayers)
net.load_state_dict(state)
net.cuda(args.gpu_id)

def close_logger(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

# slow implementation
def test(tset):
    net.eval()

    if tset=='test':
        loader = test_loader
    elif tset=='val':
        loader = val_loader
    elif tset=='train':
        loader = train_loader
    else:
        print("not recognized set in test function")

    logger.info('Testing ' + tset + ' set.')

    tdecs = []
    transcrs = []
    imfiles = []
    for (img, transcr, imfile) in loader:
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o = net(img)
        tdec = o.argmax(2).permute(1, 0).cpu().numpy()
        tdecs += [tdec]
        transcrs += list(transcr)
        imfiles += list(imfile)

    tdecs = np.concatenate(tdecs)

    rfile = open('test_results.html','w', encoding="utf-8")  
    rfile.write('<html>\n')
    rfile.write('<head><title>Squeeze Recognition</title></head>\n')
    rfile.write('<body>\n')
    rfile.write('<h1>Squeeze Recognition Results</h1><hr>\n')
        
    cer, wer = [], []
    cntc, cntw = 0, 0
    for tdec, transcr, imfile in zip(tdecs, transcrs, imfiles):
        transcr = transcr.strip()
        if transcr == 'TAEKTWN':
            print('pause')
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()

        # calculate CER and WER
        cc = float(editdistance.eval(dec_transcr, transcr))
        ww = float(editdistance.eval(dec_transcr.split(' '), transcr.split(' ')))
        cntc += len(transcr)
        cntw +=  len(transcr.split(' '))
        cer += [cc]
        wer += [ww]
        
        logger.info('Test: {} for {} ({}%)'.format(dec_transcr,transcr,max(0,100*(1-np.float64(cc)/len(transcr)))))
        rfile.write('<p><img src="{}" style="max-width:7.5in"></p>\n<p>{} [{}] ({})</p><hr>\n'.format(imfile,dec_transcr,transcr,imfile))

    cer = sum(cer) / cntc
    wer = sum(wer) / cntw

    logger.info('CER on %sset: %f', tset, cer)
    logger.info('WER on %s set: %f', tset, wer)
    #writer.add_scalar('CER', float(cer), epoch)
    #writer.add_scalar('WER', float(wer), epoch)


logger.info('Testing:')
test('val')
close_logger(logger)