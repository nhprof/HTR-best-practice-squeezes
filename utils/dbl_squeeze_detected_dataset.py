import io, os
import numpy as np 
import torch
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


class DoubleSqueezeDetectedDataset(WordLineDataset):
    def __init__(self, basefolder = 'C:/Research/Squeezes', subset = 'train', segmentation_level = 'line', 
                 fixed_size = None, transforms = None, tfprob = 0.5, character_classes = None, det_dir = '{}/Letter Boxes',
                 line_dir = '{}/Line Images'):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, tfprob, character_classes)
        self.setname = 'DblSqueezeIAS'
        self.training_lines = 'DblSqueezeTrain.xml'
        self.test_lines = 'DblSqueezeTest.xml'
        if segmentation_level == 'line':
            #self.root_dir = '{}/{}/linelevel'.format(basefolder, self.setname)
            self.root_dir = basefolder
            self.det_dir = det_dir.format(basefolder)
            self.line_dir = line_dir.format(basefolder)
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
        self.nlayers = 2;  # depth of input to network; two grayscale images in this case
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
        
        def readDetectFile(fname):
            file = open(fname,'r')
            lines = file.readlines()
            file.close()
            return [line.replace('\n','') for line in lines[:lines.index('\n')]]
        
        html = open('dataset.html','w', encoding="utf-8")
        html.write('<html>\n')
        html.write('<head><title>Squeeze Dataset</title></head>\n')
        html.write('<body>\n')
        html.write('<h1>Squeeze Dataset Visualization</h1><hr>\n')
        data = []
        det_files = [f for f in listdir(self.det_dir) if isfile(join(self.det_dir, f))]
        for dfile1 in det_files:
            if not dfile1[-12:] == '_letters.txt':
                print('Skipping nonstandard file name: '+dfile1)
                continue
            if 'Rotation1' in dfile1:
                dfile2 = dfile1.replace('Rotation1','Rotation2')
            else:
                # either rotation 2 or a non-actionable file
                continue
            if getBase(dfile1) in test_set:
                print('Skipping test set squeeze '+dfile1)
                continue
            if (subset == 'train') and getBase(dfile1) in valid_set:
                print('Reserving validation set squeeze '+dfile1)
                continue            
            if (subset != 'train') and not getBase(dfile1) in valid_set:
                continue             

            t1 = readDetectFile(join(self.det_dir,dfile1))  
            t2 = readDetectFile(join(self.det_dir,dfile2))  
            base1 = dfile1[:-4]
            base2 = dfile2[:-4]
            #base2pos = dfile.find('_Rotation')
            #base2 = dfile[:base2pos]
            if len(t1)!=len(t2):
                print('Transcript length mismatch in {}:  {} vs. {}'.format(dfile1,len(t1),len(t2)))
            for index in range(min(len(t1),len(t2))):
                # Older train/test split based upon line index, replaced by full-document lists above
                # if (subset == 'train') and (index % 5 == 2):
                #     continue
                # if (subset != 'train') and (index % 5 != 2):
                #     continue  
                if t1[index]!=t2[index]:
                    print('Transcript mismatch in {} line {}:  {} vs. {}'.format(dfile1,index,t1[index],t2[index]))
                lfname1 = join(self.line_dir,base1+'_line{0:03d}.png').format(index)
                lfname2 = join(self.line_dir,base2+'_line{0:03d}.png').format(index)
                if not isfile(lfname1):                    
                    print('{}: Unable to find line image file {}\n'.format(dfile1,lfname1))
                    continue
                if not isfile(lfname2):                    
                    print('{}: Unable to find line image file {}\n'.format(dfile2,lfname2))
                    continue
                try:
                    line_img1 = img_io.imread(lfname1)
                    line_img2 = img_io.imread(lfname2)
                except:
                    print('Problem with image files: '+lfname1+", "+lfname2)    
                    continue
                line_img1 = makeGray(1 - line_img1.astype(np.float32) / 255.0)
                line_img2 = makeGray(1 - line_img2.astype(np.float32) / 255.0)
                data.append(((line_img1, line_img2), t1[index].replace('*','').strip(), lfname1))
                html.write('<p><img src="{}"><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(lfname1,lfname2,t1[index].replace('*','').strip(),lfname1,len(data)))
                    
        html.write('</body>\n</html>\n')
        html.close()
        return(data)

        
    # def main_loader(self, subset, level) -> list:
    #     ##########################################
    #     # Load pairs of (image, ground truth)
    #     ##########################################
    #     assert(level == 'line')
        
    #     def compare(a, b):
    #         for x, y in zip(a, b):
    #             if x != y and y != '*':
    #                 return False
    #         return True
        
    #     def check(s):
    #         for c in s:
    #             if "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(c) == -1:
    #                 return False
    #         return True
        
    #     def makeGray(img):
    #         if (len(img.shape)>2):
    #             img = rgb2gray(img[:,:,:3]);
    #         return img
        
    #     out = open('transcript.txt','w', encoding="utf-8")  
    #     html = open('dataset.html','w', encoding="utf-8")
    #     html.write('<html>\n')
    #     html.write('<head><title>Squeeze Dataset</title></head>\n')
    #     html.write('<body>\n')
    #     html.write('<h1>Squeeze Dataset Visualization</h1><hr>\n')
    #     data = []
    #     all_lines_parsed = 0
    #     all_problem_lines = 0
    #     all_lines_failed_to_open = 0
    #     anno_files = [f for f in listdir(self.anno_dir) if isfile(join(self.anno_dir, f))]
    #     for f in anno_files:
    #         lines_parsed = 0
    #         problem_lines = 0
    #         lines_failed_to_open = 0
    #         fdata = pd.read_excel(join(self.anno_dir, f))
    #         for index, row in fdata.iterrows():
    #             if (subset == 'train') and (index % 5 == 2):
    #                 continue
    #             if (subset != 'train') and (index % 5 != 2):
    #                 continue      
    #             try:
    #                 lf1 = row.iloc[0]
    #                 lf2 = row.iloc[1]
    #                 phi_tag = row.iloc[2]
    #                 hand_tag = row.iloc[3]
    #             except:
    #                 out.write('{}: Not enough values values found in a row\n'.format(f))
    #                 lines_failed_to_open += 1
    #             #print("Reading {} line {} {} {} {}".format(f,lf1,lf2,phi_tag,hand_tag))
    #             if not isinstance(lf1,str):
    #                 out.write('Warning. {}, {}: Non-string value found in first column\n'.format(lf1, lf2))
    #                 problem_lines += 1
    #                 lf1 = ''
    #             if not isinstance(lf2,str):
    #                 out.write('Warning. {}, {}: Non-string value found in second column\n'.format(lf1, lf2))
    #                 problem_lines += 1
    #                 lf2 = ''
    #             if (not isinstance(phi_tag,str) or not isinstance(hand_tag,str)):
    #                 out.write('{}, {}: Non-string values found in tags\n'.format(lf1, lf2))
    #                 lines_failed_to_open += 1
    #                 continue
    #             if lf1[-4:]!='.png' and lf1[-4:]!='.PNG':
    #                 lf1 = lf1+'.png'
    #             if lf2[-4:]!='.png' and lf2[-4:]!='.PNG':
    #                 lf2 = lf2+'.png'
    #             phi_tag = phi_tag.upper().replace(' ','');
    #             hand_tag = hand_tag.upper().replace(' ','');
    #             if not check(phi_tag):
    #                 out.write('{}: Invalid characters found in column 3: {}\n'.format(lf1,phi_tag))
    #                 lines_failed_to_open += 1                    
    #             else: 
    #                 if (len(phi_tag) != len(hand_tag)):
    #                     out.write('{}: Warning -- Tag length mismatch - `{}` vs. `{}`\n'.format(lf1,phi_tag,hand_tag))
    #                     problem_lines += 1
    #                 if compare(phi_tag,hand_tag) == False:
    #                     out.write('{}: Warning -- Tag character mismatch - `{}` vs. `{}`\n'.format(lf1,phi_tag,hand_tag))
    #                     problem_lines += 1
    #                 try:
    #                     f1 = join(join(self.line_dir,lf1[0:lf1.find('300dpi')+6]),lf1)
    #                 except:
    #                     out.write('{}: Problem with line image file name {}\n'.format(f,lf1))
    #                     lines_failed_to_open += 1
    #                     continue   
    #                 if isfile(f1):
    #                     try:
    #                         line_img1 = img_io.imread(f1)
    #                     except:
    #                         out.write('{}: Problem loading line image file {}\n'.format(f,f1))
    #                         lines_failed_to_open += 1
    #                         continue
    #                     line_img1 = makeGray(1 - line_img1.astype(np.float32) / 255.0)
    #                     #data.append(
    #                     #    (line_img, hand_tag.replace('*','').strip(), f1)
    #                     #)
    #                     html.write('<p><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(f1,hand_tag.replace('*','').strip(),f1,len(data)))
    #                     lines_parsed += 1
    #                 else:
    #                     out.write('{}: Unable to find line image file {}\n'.format(f,f1))
    #                     lines_failed_to_open += 1

    #                 try:
    #                     f2 = join(join(self.line_dir,lf2[0:lf2.find('300dpi')+6]),lf2)
    #                 except:
    #                     out.write('{}: Problem with line image file name {}\n'.format(f,lf2))
    #                     lines_failed_to_open += 1
    #                     continue                            
    #                 if isfile(f2):
    #                     try:
    #                         line_img2 = img_io.imread(f2)
    #                     except:
    #                         out.write('{}: Problem loading line image file {}\n'.format(f,f2))
    #                         lines_failed_to_open += 1
    #                         continue
    #                     line_img2 = makeGray(1 - line_img2.astype(np.float32) / 255.0)
    #                     data.append(
    #                         ((line_img1,line_img2), hand_tag.replace('*','').strip(), f2)
    #                     )
    #                     html.write('<p><img src="{}"></p>\n<p>{} {} (#{})</p><hr>\n'.format(f2,hand_tag.replace('*','').strip(),f2,len(data)))
    #                     lines_parsed += 1
    #                 else:
    #                     out.write('{}: Unable to find line image file {}\n'.format(f,f2))
    #                     lines_failed_to_open += 1
    #                 #print(lines_parsed)
    #         all_lines_parsed += lines_parsed
    #         all_problem_lines += problem_lines
    #         all_lines_failed_to_open += lines_failed_to_open
    #         out.write('Read {} lines successfully from {}. {} problems noted.  {} lines failed to load.\n\n'.format(lines_parsed, f, problem_lines, lines_failed_to_open))
    #     out.write('Read {} lines successfully in total. {} problems noted.  {} lines failed to load.\n'.format(all_lines_parsed, all_problem_lines, all_lines_failed_to_open))
    #     out.close()
    #     html.write('</body>\n</html>\n')
    #     html.close()
    #     return(data)
    
    # modified version of word_dataset's method, to process two images at once.
    def __getitem__(self, index):
        img1 = self.data[index][0][0]
        img2 = self.data[index][0][1]
        transcr = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        if self.subset == 'train':
            nwidth1 = int(np.random.uniform(.75, 1.25) * img1.shape[1])
            nheight1 = int((np.random.uniform(.9, 1.1) * img1.shape[0] / img1.shape[1]) * nwidth1)
            #nwidth2 = int(np.random.uniform(.75, 1.25) * img2.shape[1])
            #nheight2 = int((np.random.uniform(.9, 1.1) * img2.shape[0] / img2.shape[1]) * nwidth2)
            nwidth2 = nwidth1
            nheight2 = nwidth2
        else:
            nheight1, nwidth1 = img1.shape[0], img1.shape[1]
            nheight2, nwidth2 = img2.shape[0], img2.shape[1]

        nheight1, nwidth1 = max(4, min(fheight-16, nheight1)), max(8, min(fwidth-32, nwidth1))
        nheight2, nwidth2 = max(4, min(fheight-16, nheight2)), max(8, min(fwidth-32, nwidth2))
        img1 = image_resize(img1, height=int(1.0 * nheight1), width=int(1.0 * nwidth1))
        img2 = image_resize(img2, height=int(1.0 * nheight2), width=int(1.0 * nwidth2))

        img1 = centered(img1, (fheight, fwidth), border_value=None)
        img2 = centered(img2, (fheight, fwidth), border_value=None)
        ims = np.stack((img1,img2),0)
        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    ims = tr(ims)
        # pad with zeroes
        #img = centered(img, (fheight, fwidth), np.random.uniform(.2, .8, size=2), border_value=0.0)
        ims = torch.Tensor(ims).float()
        return ims, transcr, *self.data[index][2:]  # pass through extra data id supplied