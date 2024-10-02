import io, os
import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset, LineListIO
from os.path import isfile
from utils.auxilary_functions import image_resize, centered
from skimage.color import rgb2gray
from skimage.transform import resize
import tqdm
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image
import xml.etree.ElementTree as ET
import sys
sys.path.append('../Squeezes')
from BoxLabels import Box, readBoxFile

valid_set = [
    'IG_II2_7_Copy1',
    'IG_II2_61',
    'IG_II2_149',
    'IG_II2_209_Copy1',
    'IG_II2_264',
    'IG_II2_336b',
    'IG_II2_359',
    'IG_II2_414b',
    'IG_II2_478a',
    'IG_II2_511_Copy2',
    'IG_II2_1675_Copy2',
    'IG_II2_1690',
    'IG_II2_1708',
    'IG_II2_1801_Copy2',
    'IG_II2_1932',
    'IG_II2_1976_Copy2',
    'IG_II2_1995',
    'IG_II2_2019',
    'IG_II2_2034_Copy2',
    'IG_II2_2051A_Merged',
    'IG_II2_2073b',
    'IG_II2_2089e',
    'IAS_27253',
    'IAS_27281',
    'IAS_27397',
    'IAS_27743',
    'IAS23594',
    'IAS24096',
    '⁮IAS24407',
    'IAS33147',
    ]

test_set = [
    'IG_II2_42_Copy1',
    'IG_II2_134',
    'IG_II2_185_Copy2',
    'IG_II2_230a',
    'IG_II2_333a+b',
    'IG_II2_350',
    'IG_II2_398b',
    'IG_II2_454',
    'IG_II2_493_Copy1',
    'IG_II2_1682b_Copy1',
    'IG_II2_1701',
    'IG_II2_1723',
    'IG_II2_1821',
    'IG_II2_1952b',
    'IG_II2_1979',
    'IG_II2_2005_Copy1',
    'IG_II2_2024(Left)',
    'IG_II2_2041A_Merged',
    'IG_II2_2061b+2093b',
    'IG_II2_2089a',
    'IG_II2_2155',
    'IAS_27272',
    'IAS_27319',
    'IAS_27613',
    'IAS23566',
    '⁮IAS24039',
    '⁮IAS24371',
    'IAS33120_Merged',
    'IAS33163',
    ]

class SqueezeDataset(WordLineDataset):
    def __init__(self, basefolder = 'C:/Research/Squeezes', subset = 'train', segmentation_level = 'line', 
                 fixed_size = None, transforms = None, tfprob = 0.5, character_classes = None, box_dir = '{}/Letter Boxes',
                 line_dir = '{}/LB Lines', aux_dir = None):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, tfprob, character_classes)
        self.setname = 'SqueezeIAS'
        self.training_lines = 'SqueezeTrain.xml'
        self.test_lines = 'SqueezeTest.xml'
        if segmentation_level == 'line':
            #self.root_dir = '{}/{}/linelevel'.format(basefolder, self.setname)
            self.root_dir = basefolder
            self.box_dir = box_dir.format(basefolder)
            self.line_dir = line_dir.format(basefolder)
            if aux_dir == None:
                self.aux_dir = None
            else:
                self.aux_dir = aux_dir.format(basefolder)
            if subset == 'train':
                self.xmlfile = '{}/{}'.format(self.root_dir, self.training_lines)
            elif subset == 'test':
                self.xmlfile = '{}/{}'.format(self.root_dir, self.test_lines)
            else:
                raise ValueError('partition must be one of None, train or test')
        elif segmentation_level == 'word':
            raise ValueError('Word level segmentation not available.')
        else:
            raise ValueError('Segmentation level must be either word or line level.')
        self.data = [] # A list of tuples (image data, transcription)
        self.query_list = None
        self.dictionary_file = '' #Unused in this version
        self.nlayers = 1;  # depth of input to network; grayscale image in this case
        super().__finalize__()

    def main_loader(self, subset, level) -> list:
        ##########################################
        # Load pairs of (image, ground truth)
        ##########################################
        assert(level == 'line')
        
        def compare(a, b):
            for x, y in zip(a, b):
                if x != y and y != '*':
                    return False
            return True
        
        def check(s):
            for c in s:
                if "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(c) == -1:
                    return False
            return True
        
        def makeGray(img):
            if (len(img.shape)>2):
                img = rgb2gray(img[:,:,:3]);
            return img
        
        def getBase(s):
            irot = s.find('_Rotation')
            imer = s.find('_Merged')
            return s[:min(irot%len(s),imer%len(s))]
        
        html = open('dataset.html','w', encoding="utf-8")
        html.write('<html>\n')
        html.write('<head><title>Squeeze Dataset</title></head>\n')
        html.write('<body>\n')
        html.write('<h1>Squeeze Dataset Visualization</h1><hr>\n')
        data = []
        box_files = [f for f in listdir(self.box_dir) if isfile(join(self.box_dir, f))]
        for bfile in box_files:
            if not bfile[-12:] == '_letters.txt':
                print('Skipping nonstandard file name: '+bfile)
                continue
            if getBase(bfile) in test_set:
                print('Skipping test set squeeze '+bfile)
                continue
            if (subset == 'train') and getBase(bfile) in valid_set:
                print('Reserving validation set squeeze '+bfile)
                continue            
            if (subset != 'train') and not getBase(bfile) in valid_set:
                continue  
            a,b,t,l = readBoxFile(join(self.box_dir,bfile))  
            base = bfile[:-12]
            #base2pos = bfile.find('_Rotation')
            #base2 = bfile[:base2pos]
            for index in range(len(t)):
                # Older train/test split based upon line index, replaced by full-document lists above
                # if (subset == 'train') and (index % 5 == 2):
                #     continue
                # if (subset != 'train') and (index % 5 != 2):
                #     continue  
                lfname = join(self.line_dir,base+'_line{0:03d}.png').format(index)
                if not isfile(lfname):                    
                    print('{}: Unable to find line image file {}\n'.format(bfile,lfname))
                    continue
                try:
                    line_img = img_io.imread(lfname)
                except:
                    print('Problem with image file: '+lfname)    
                    continue
                line_img = makeGray(1 - line_img.astype(np.float32) / 255.0)
                data.append((line_img, t[index].replace('*','').strip(), lfname))
                html.write('<p><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(lfname,t[index].replace('*','').strip(),lfname,len(data)))
                if not self.aux_dir == None:
                    # add auxiliary line image
                    alfname = join(self.aux_dir,base+'_line{0:03d}.png').format(index)
                    if not isfile(lfname):                    
                        print('{}: Unable to find line image file {}\n'.format(bfile,alfname))
                        continue
                    try:
                        line_img = img_io.imread(alfname)
                    except:
                        print('Problem with image file: '+alfname)    
                        continue
                    line_img = makeGray(1 - line_img.astype(np.float32) / 255.0)
                    data.append((line_img, t[index].replace('*','').strip(), lfname))
                    html.write('<p><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(alfname,t[index].replace('*','').strip(),lfname,len(data)))

        html.write('</body>\n</html>\n')
        html.close()
        return(data)


    # this version of the main loader uses the annotation files instead of the letter box files.
    def old_main_loader(self, subset, level) -> list:
        ##########################################
        # Load pairs of (image, ground truth)
        ##########################################
        assert(level == 'line')
        
        def compare(a, b):
            for x, y in zip(a, b):
                if x != y and y != '*':
                    return False
            return True
        
        def check(s):
            for c in s:
                if "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(c) == -1:
                    return False
            return True
        
        def makeGray(img):
            if (len(img.shape)>2):
                img = rgb2gray(img[:,:,:3]);
            return img
        
        out = open('transcript.txt','w', encoding="utf-8")  
        html = open('dataset.html','w', encoding="utf-8")
        html.write('<html>\n')
        html.write('<head><title>Squeeze Dataset</title></head>\n')
        html.write('<body>\n')
        html.write('<h1>Squeeze Dataset Visualization</h1><hr>\n')
        data = []
        all_lines_parsed = 0
        all_problem_lines = 0
        all_lines_failed_to_open = 0
        anno_files = [f for f in listdir(self.anno_dir) if isfile(join(self.anno_dir, f))]
        for f in anno_files:
            lines_parsed = 0
            problem_lines = 0
            lines_failed_to_open = 0
            fdata = pd.read_excel(join(self.anno_dir, f), header=None)
            for index, row in fdata.iterrows():
                if (subset == 'train') and (index % 5 == 2):
                    continue
                if (subset != 'train') and (index % 5 != 2):
                    continue      
                try:
                    lf1 = row.iloc[0]
                    lf2 = row.iloc[1]
                    phi_tag = row.iloc[2]
                    hand_tag = row.iloc[3]
                except:
                    out.write('{}: Not enough values values found in a row\n'.format(f))
                    lines_failed_to_open += 1
                    continue
                #print("Reading {} line {} {} {} {}".format(f,lf1,lf2,phi_tag,hand_tag))
                if not isinstance(lf1,str):
                    out.write('Warning. {}, {}: Non-string value found in first column\n'.format(lf1, lf2))
                    problem_lines += 1
                    lf1 = ''
                if not isinstance(lf2,str):
                    out.write('Warning. {}, {}: Non-string value found in second column\n'.format(lf1, lf2))
                    problem_lines += 1
                    lf2 = ''
                if (not isinstance(phi_tag,str) or not isinstance(hand_tag,str)):
                    out.write('{}, {}: Non-string values found in tags\n'.format(lf1, lf2))
                    lines_failed_to_open += 1
                    continue
                if lf1[-4:]!='.png' and lf1[-4:]!='.PNG':
                    lf1 = lf1+'.png'
                if lf2[-4:]!='.png' and lf2[-4:]!='.PNG':
                    lf2 = lf2+'.png'
                phi_tag = phi_tag.upper().replace(' ','');
                hand_tag = hand_tag.upper().replace(' ','');
                if not check(phi_tag):
                    out.write('{}: Invalid characters found in column 3: {}\n'.format(lf1,phi_tag))
                    lines_failed_to_open += 1     
                    continue
                if (len(phi_tag) != len(hand_tag)):
                    out.write('{}: Warning -- Tag length mismatch - `{}` vs. `{}`\n'.format(lf1,phi_tag,hand_tag))
                    problem_lines += 1
                if compare(phi_tag,hand_tag) == False:
                    out.write('{}: Warning -- Tag character mismatch - `{}` vs. `{}`\n'.format(lf1,phi_tag,hand_tag))
                    problem_lines += 1
                    
                # Read rotation 1 line file:
                try:
                    # first try direct lookup of file from lb2lbl:
                    f1 = join(self.line_dir,f[0:f.find('.xlsx')]+'_Rotation1_300dpi_line{0:03d}.png'.format(index))
                    if not isfile(f1):                    
                        # use older way: name from annotation file & look in subfolder
                        f1 = join(join(self.line_dir,lf1[0:lf1.find('300dpi')+6]),lf1)
                    if not isfile(f1):                    
                        out.write('{}: Unable to find line image file {}\n'.format(f,f1))
                        lines_failed_to_open += 1
                        continue
                    line_img = img_io.imread(f1)
                    line_img = makeGray(1 - line_img.astype(np.float32) / 255.0)
                    data.append((line_img, hand_tag.replace('*','').strip(), f1))
                    html.write('<p><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(f1,hand_tag.replace('*','').strip(),f1,len(data)))
                    lines_parsed += 1
                except:
                    out.write('{}: Problem with line image file name {}\n'.format(f,lf1))
                    lines_failed_to_open += 1
                    continue   

                # Read rotation 2 line file:
                try:
                    # first try direct lookup of file from lb2lbl:
                    f2 = join(self.line_dir,f[0:f.find('.xlsx')]+'_Rotation2_300dpi_line{0:03d}.png'.format(index))
                    if not isfile(f2):                    
                        # use older way: name from annotation file & look in subfolder
                        f2 = join(join(self.line_dir,lf2[0:lf2.find('300dpi')+6]),lf2)
                    if not isfile(f2):                    
                        out.write('{}: Unable to find line image file {}\n'.format(f,f2))
                        lines_failed_to_open += 1
                        continue
                    line_img = img_io.imread(f2)
                    line_img = makeGray(1 - line_img.astype(np.float32) / 255.0)
                    data.append((line_img, hand_tag.replace('*','').strip(), f2))
                    html.write('<p><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(f2,hand_tag.replace('*','').strip(),f2,len(data)))
                    lines_parsed += 1
                except:
                    out.write('{}: Problem with line image file name {}\n'.format(f,lf2))
                    lines_failed_to_open += 1
                    continue   
            all_lines_parsed += lines_parsed
            all_problem_lines += problem_lines
            all_lines_failed_to_open += lines_failed_to_open
            out.write('Read {} lines successfully from {}. {} problems noted.  {} lines failed to load.\n\n'.format(lines_parsed, f, problem_lines, lines_failed_to_open))
        out.write('Read {} lines successfully in total. {} problems noted.  {} lines failed to load.\n'.format(all_lines_parsed, all_problem_lines, all_lines_failed_to_open))
        out.close()
        html.write('</body>\n</html>\n')
        html.close()
        return(data)