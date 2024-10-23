import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
from os.path import isfile
from skimage import io
from torchvision.utils import save_image
from skimage.transform import resize
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
from utils.iam_dataset import IAMDataset
from utils.auxilary_functions import affine_transformation
from feature_extractor import ImageEncoder
import timm
import cv2
import time
import json
import random


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class WordStyleDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)   
    '''
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    '''
    def __init__(self, 
        basefolder: str = 'datasets/',                #Root folder
        subset: str = 'all',                          #Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = 'line',             #Type of data to load ('line' or 'word')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        ):
        
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None                             # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.max_transcr_len = 0
        self.data_file = './iam_data/iam_train_val_fixed.txt'

        with open(self.data_file, 'r') as f:
            lines = f.readlines()
        
        self.data_info = [line.strip().split(',') for line in lines]
        
    def __len__(self):
        return len(self.data_info)

   
    def __getitem__(self, index):
        
        img = self.data_info[index][0]
        img = Image.open(img).convert('RGB')
        transcr = self.data_info[index][2]

        wid = self.data_info[index][1]

        img_path = self.data_info[index][0]
        #pick another sample that has the same self.data[2] or same writer id
        positive_samples = [p for p in self.data_info if p[1] == wid and len(p[2])>3]
        negative_samples = [n for n in self.data_info if n[1] != wid and len(n[2])>3]
        
        #print('wid', wid)
        positive = random.choice(positive_samples)[0]
        
        #print('positive', positive)
        #pick another image from a different writer
        negative = random.choice(negative_samples)[0]
        #print('negative', negative)
        img_pos = Image.open(positive).convert('RGB') #image_resize_PIL(positive, height=positive.height // 2)
        img_neg = Image.open(negative).convert('RGB') #image_resize_PIL(negative, height=negative.height // 2)
        
        if img.height < 64 and img.width < 256:
            img = img
        else:
            img = image_resize_PIL(img, height=img.height // 2)
        
        if img_pos.height < 64 and img_pos.width < 256:
            img_pos = img_pos
        else:
            img_pos = image_resize_PIL(img_pos, height=img_pos.height // 2)
        
        if img_neg.height < 64 and img_neg.width < 256:
            img_neg = img_neg
        else:
            img_neg = image_resize_PIL(img_neg, height=img_neg.height // 2)
        
        
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        #print('fheight', fheight, 'fwidth', fwidth)
        if self.subset == 'train':
            nwidth = int(np.random.uniform(.75, 1.25) * img.width)
            nheight = int((np.random.uniform(.9, 1.1) * img.height / img.width) * nwidth)
            
            nwidth_pos = int(np.random.uniform(.75, 1.25) * img_pos.width)
            nheight_pos = int((np.random.uniform(.9, 1.1) * img_pos.height / img_pos.width) * nwidth_pos)
            
            nwidth_neg = int(np.random.uniform(.75, 1.25) * img_neg.width)
            nheight_neg = int((np.random.uniform(.9, 1.1) * img_neg.height / img_neg.width) * nwidth_neg)
            
        else:
            nheight, nwidth = img.height, img.width
            nheight_pos, nwidth_pos = img_pos.height, img_pos.width
            nheight_neg, nwidth_neg = img_neg.height, img_neg.width
            
        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        nheight_pos, nwidth_pos = max(4, min(fheight-16, nheight_pos)), max(8, min(fwidth-32, nwidth_pos))
        nheight_neg, nwidth_neg = max(4, min(fheight-16, nheight_neg)), max(8, min(fwidth-32, nwidth_neg))
        
        img = image_resize_PIL(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        img = centered_PIL(img, (fheight, fwidth), border_value=255.0)
       
        img_pos = image_resize_PIL(img_pos, height=int(1.0 * nheight_pos), width=int(1.0 * nwidth_pos))
        img_pos = centered_PIL(img_pos, (fheight, fwidth), border_value=255.0)
        
        img_neg = image_resize_PIL(img_neg, height=int(1.0 * nheight_neg), width=int(1.0 * nwidth_neg))
        img_neg = centered_PIL(img_neg, (fheight, fwidth), border_value=255.0)
        
        
        if self.transforms is not None:
            
            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)
        
        
        return img, transcr, wid, img_pos, img_neg, img_path

    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        img, transcr, wid, positive, negative, img_path = zip(*batch)

        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        #transcr_batch = torch.stack(transcr)
        #char_tokens_batch = torch.stack(char_tokens)
        
        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)
        
        
        return images_batch, transcr, wid, images_pos, images_neg, img_path


def image_resize_PIL(img, height=None, width=None):
    if height is None and width is None:
        return img  # No resizing needed

    original_width, original_height = img.size

    if height is not None and width is None:
        scale = height / original_height
        new_width = int(original_width * scale)
        new_height = height
    elif width is not None and height is None:
        scale = width / original_width
        new_width = width
        new_height = int(original_height * scale)
    else:
        new_width = width
        new_height = height

    # Resize the image
    resized_img = img.resize((new_width, new_height))
    #resized_img.save('res.png')
    return resized_img


def centered_PIL(word_img, tsize, centering=(.5, .5), border_value=None):
    
    height = tsize[0]
    width = tsize[1]
    #print('word_img.size', word_img.size)
    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.height
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.height - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.width
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.width - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    
    
   
    #print('word_img.size, padw, padh', word_img.size, padw, padh)
    res = Image.new('RGB', (width, height), color = (255, 255, 255))
    #res.save('background.png')
    
    res.paste(word_img, (padw[0], padh[0]))
    
    
    return res

class WordLineDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)   
    '''
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    '''
    def __init__(self, 
        basefolder: str = 'datasets/',                #Root folder
        subset: str = 'all',                          #Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = 'line',             #Type of data to load ('line' or 'word')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        ):
        
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None                             # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes
        self.max_transcr_len = 0
        #self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", )

    def __finalize__(self):
        '''
        Will call code after descendant class has specified 'key' variables
        and ran dataset-specific code
        '''
        assert(self.setname is not None)
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(','))
            self.stopwords = self.stopwords[0]
        
        save_path = './IAM_dataset_PIL_style'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        save_file = '{}/{}_{}_{}.pt'.format(save_path, self.subset, self.segmentation_level, self.setname) #dataset_path + '/' + set + '_' + level + '_IAM.pt'
        print('save_file', save_file)
        #if isfile(save_file) is False:
        #    data = self.main_loader(self.subset, self.segmentation_level)
        #    torch.save(data, save_file)   #Uncomment this in 'release' version
        #else:
        #    data = torch.load(save_file)
        
        data = self.main_loader(self.subset, self.segmentation_level)
        self.data = data
        #print('data', self.data)
        self.initial_writer_ids = [d[2] for d in data]
        
        writer_ids,_  = np.unique([d[2] for d in data], return_inverse=True)
       
        self.writer_ids = writer_ids
        
        self.wclasses = len(writer_ids)
        print('Number of writers', self.wclasses)
        if self.character_classes is None:
            res = set()
             #compute character classes given input transcriptions
            for _,transcr,_,_ in tqdm(data):
                #print('legth transcr = ', len(transcr))
                res.update(list(transcr))
                self.max_transcr_len = max(self.max_transcr_len, len(transcr))
                #print('self.max_transcr_len', self.max_transcr_len)
                
            res = sorted(list(res))
            res.append(' ')
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            print('Max transcription length: {}'.format(self.max_transcr_len))
            self.character_classes = res
            self.max_transcr_len = self.max_transcr_len
        #END FINALIZE

    def __len__(self):
        return len(self.data)

   
    def __getitem__(self, index):
        
        img = self.data[index][0]
        
        transcr = self.data[index][1]

        wid = self.data[index][2]

        img_path = self.data[index][3]
        #pick another sample that has the same self.data[2] or same writer id
        positive_samples = [p for p in self.data if p[2] == wid and len(p[1])>3]
        negative_samples = [n for n in self.data if n[2] != wid and len(n[1])>3]
        
        
        positive = random.choice(positive_samples)[0]
        
        # Make sure you have at least 5 matching images
        if len(positive_samples) >= 5:
            # Randomly select 5 indices from the matching_indices
            random_samples = random.sample(positive_samples, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples]
        else:
            # Handle the case where there are fewer than 5 matching images (if needed)
            #print("Not enough matching images with writer ID", wid)
            positive_samples_ = [p for p in self.data if p[2] == wid]
            #print('len positive samples', len(positive_samples_), 'wid', wid)
            random_samples_ = random.sample(positive_samples_, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples_]

        #pick another image from a different writer
        negative = random.choice(negative_samples)[0]

        img_pos = positive #image_resize_PIL(positive, height=positive.height // 2)
        img_neg = negative #image_resize_PIL(negative, height=negative.height // 2)
        
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        #print('fheight', fheight, 'fwidth', fwidth)
        if self.subset == 'train':
            nwidth = int(np.random.uniform(.75, 1.25) * img.width)
            nheight = int((np.random.uniform(.9, 1.1) * img.height / img.width) * nwidth)
            
            nwidth_pos = int(np.random.uniform(.75, 1.25) * img_pos.width)
            nheight_pos = int((np.random.uniform(.9, 1.1) * img_pos.height / img_pos.width) * nwidth_pos)
            
            nwidth_neg = int(np.random.uniform(.75, 1.25) * img_neg.width)
            nheight_neg = int((np.random.uniform(.9, 1.1) * img_neg.height / img_neg.width) * nwidth_neg)
            
        else:
            nheight, nwidth = img.height, img.width
            nheight_pos, nwidth_pos = img_pos.height, img_pos.width
            nheight_neg, nwidth_neg = img_neg.height, img_neg.width
            
        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        nheight_pos, nwidth_pos = max(4, min(fheight-16, nheight_pos)), max(8, min(fwidth-32, nwidth_pos))
        nheight_neg, nwidth_neg = max(4, min(fheight-16, nheight_neg)), max(8, min(fwidth-32, nwidth_neg))
        
        #img = image_resize_PIL(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        #img = centered_PIL(img, (fheight, fwidth), border_value=None).convert('L')
        
            #image = image.resize((256, 64), Image.ANTIALIAS)
        if img.width < 256:
            img = ImageOps.pad(img, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
        #print('img', img.mode, img.size)
        
        pixel_values_img = img #self.processor(img, return_tensors="pt").pixel_values
        pixel_values_img = pixel_values_img#.squeeze(0)

        img_pos = image_resize_PIL(img_pos, height=int(1.0 * nheight_pos), width=int(1.0 * nwidth_pos))
        img_pos = centered_PIL(img_pos, (fheight, fwidth), border_value=255.0)
        
        img_neg = image_resize_PIL(img_neg, height=int(1.0 * nheight_neg), width=int(1.0 * nwidth_neg))
        img_neg = centered_PIL(img_neg, (fheight, fwidth), border_value=255.0)
        
        pixel_values_pos = img_pos #self.processor(img_pos, return_tensors="pt").pixel_values
        pixel_values_neg = img_neg #self.processor(img_neg, return_tensors="pt").pixel_values
        pixel_values_pos = pixel_values_pos#.squeeze(0)
        
        pixel_values_neg = pixel_values_neg#.squeeze(0)
        
        st_imgs = []
        for s_im in style_images:
            #s_im = image_resize_PIL(s_im, height=s_im.height // 2)
            if self.subset == 'train':
                nwidth = int(np.random.uniform(.75, 1.25) * s_im.width)
                nheight = int((np.random.uniform(.9, 1.1) * s_im.height / s_im.width) * nwidth)
                
            else:
                nheight, nwidth = s_im.height, s_im.width
                
            nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
            # Load the image and transform it
            s_img = image_resize_PIL(s_im, height=int(1.0 * nheight), width=int(1.0 * nwidth))
            s_img = centered_PIL(s_img, (fheight, fwidth), border_value=255.0)
            if self.transforms is not None:
                s_img_tensor = self.transforms(img)
            
            st_imgs += [s_img_tensor]
            
        s_imgs = torch.stack(st_imgs)
        
        if self.transforms is not None:
            
            img = self.transforms(img)
            img_pos = self.transforms(img_pos)
            img_neg = self.transforms(img_neg)
        
        char_tokens = [self.character_classes.index(c) for c in transcr]
        #print('char_tokens before', char_tokens)
        pad_token = 79 
        
        #padding_length = self.max_transcr_len - len(char_tokens)
        padding_length = 95 - len(char_tokens)
        char_tokens.extend([pad_token] * padding_length)
        
        #char_tokens += [pad_token] * (self.max_transcr_len - len(char_tokens))
        char_tokens = torch.tensor(char_tokens, dtype=torch.long)
        
        cla = self.character_classes
        #print('character classes', cla)
        #wid = self.wr_dict[index]
        #print('wid after', index, wid)
        #print('pixel_values_pos', pixel_values_pos.shape)
        #img = outImg
        #save_image(img, 'check_augm.png')
        return img, transcr, char_tokens, wid, img_pos, img_neg, cla, s_imgs, img_path, img, img_pos, img_neg #pixel_values_img, pixel_values_pos, pixel_values_neg

    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        img, transcr, char_tokens, wid, positive, negative, cla, s_imgs, img_path, pixel_values_img, pixel_values_pos, pixel_values_neg = zip(*batch)

        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        #transcr_batch = torch.stack(transcr)
        char_tokens_batch = torch.stack(char_tokens)
        
        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)
        
        s_imgs = torch.stack(s_imgs)
        
        pixel_values_img = torch.stack(pixel_values_img)
        
        pixel_values_pos = torch.stack(pixel_values_pos)
        pixel_values_neg = torch.stack(pixel_values_neg)
        
        return img, transcr, char_tokens_batch, wid, images_pos, images_neg, cla, s_imgs, img_path, pixel_values_img, pixel_values_pos, pixel_values_neg

    
    
    def main_loader(self, subset, segmentation_level) -> list:
        # This function should be implemented by an inheriting class.
        raise NotImplementedError

    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        '''
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not
        
        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        '''
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError('The requested fixed image size is invalid!')
            new_img = resize(image=img, output_shape=fixed_image_size[::-1], mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print('OUCH')
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img
    
    def print_random_sample(self, image, transcription, id, as_saved_files=True):
        import random    #   Create method that will show example images using graphics-in-console (e.g. TerminalImageViewer)
        from PIL import Image
        # Run this with a very low probability
        x = random.randint(0, 10000)
        if(x > 5):
            return
        def show_image(img):
            def get_ansi_color_code(r, g, b):
                if r == g and g == b:
                    if r < 8:
                        return 16
                    if r > 248:
                        return 231
                    return round(((r - 8) / 247) * 24) + 232
                return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
            def get_color(r, g, b):
                return "\x1b[48;5;{}m \x1b[0m".format(int(get_ansi_color_code(r,g,b)))
            h = 12
            w = int((img.width / img.height) * h)
            img = img.resize((w,h))
            img_arr = np.asarray(img)
            h,w  = img_arr.shape #,c
            for x in range(h):
                for y in range(w):
                    pix = img_arr[x][y]
                    print(get_color(pix, pix, pix), sep='', end='')
                    #print(get_color(pix[0], pix[1], pix[2]), sep='', end='')
                print()
        if(as_saved_files):
            Image.fromarray(np.uint8(image*255.)).save('/tmp/a{}_{}.png'.format(id, transcription))
        else:
            print('Id = {}, Transcription = "{}"'.format(id, transcription))
            show_image(Image.fromarray(255.0*image))
            print()

class LineListIO(object):
    '''
    Helper class for reading/writing text files into lists.
    The elements of the list are the lines in the text file.
    '''
    @staticmethod
    def read_list(filepath, encoding='ascii'):        
        if not os.path.exists(filepath):
            raise ValueError('File for reading list does NOT exist: ' + filepath)
        
        linelist = []        
        if encoding == 'ascii':
            transform = lambda line: line.encode()
        else:
            transform = lambda line: line 

        with io.open(filepath, encoding=encoding) as stream:            
            for line in stream:
                line = transform(line.strip())
                if line != '':
                    linelist.append(line)                    
        return linelist

    @staticmethod
    def write_list(file_path, line_list, encoding='ascii', 
                   append=False, verbose=False):
        '''
        Writes a list into the given file object
        
        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''                
        mode = 'w'
        if append:
            mode = 'a'
        
        with io.open(file_path, mode, encoding=encoding) as f:
            if verbose:
                line_list = tqdm.tqdm(line_list)
              
            for l in line_list:
                #f.write(unicode(l) + '\n')   Python 2
                f.write(l + '\n')


class IAMDataset_style(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = 'IAM'
        self.trainset_file = '{}/{}/set_split/trainset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/set_split/validationset1.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/set_split/testset.txt'.format(self.basefolder, self.setname)
        self.line_file = '{}/ascii/lines.txt'.format(self.basefolder, self.setname)
        self.word_file = './iam_data/ascii/words.txt'.format(self.basefolder, self.setname)
        self.word_path = '{}/words'.format(self.basefolder, self.setname)
        self.line_path = '{}/lines'.format(self.basefolder, self.setname)
        self.forms = './iam_data/ascii/forms.txt'
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(self, set='train', level='word'):
            if subset == 'train':
                #valid_set = np.loadtxt(self.trainset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/train_val.uttlist', dtype=str)
                #print(valid_set)
            elif subset == 'val':
                #valid_set = np.loadtxt(self.valset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/validation.uttlist', dtype=str)
            elif subset == 'test':
                #valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/test.uttlist', dtype=str)
            else:
                raise ValueError
            if level == 'word':
                gtfile= self.word_file
                root_path = self.word_path
                print('root_path', root_path)
                forms = self.forms
            elif level == 'line':
                gtfile = self.line_file
                root_path = self.line_path
            else:
                raise ValueError
            gt = []
            form_writer_dict = {}
            
            dict_path = f'./writers_dict_{subset}.json'
            #open dict file
            with open(dict_path, 'r') as f:
                wr_dict = json.load(f)
            for l in open(forms):
                if not l.startswith("#"):
                    info = l.strip().split()
                    #print('info', info)
                    form_name = info[0]
                    writer_name = info[1]
                    form_writer_dict[form_name] = writer_name
                    #print('form_writer_dict', form_writer_dict)
                    #print('form_name', form_name)
                    #print('writer', writer_name)
            
            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]
                    name_parts = name.split('-')
                    pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
                    #print('name', name)
                    #form =
                    #writer_name = name_parts[1]
                    #print('writer_name', writer_name)
                    
                    if level == 'word':
                        line_name = pathlist[-2]
                        del pathlist[-2]

                        if (info[1] != 'ok'):
                            continue

                    elif level == 'line':
                        line_name = pathlist[-1]
                    form_name = '-'.join(line_name.split('-')[:-1])
                    #print('form_name', form_name)
                    #if (info[1] != 'ok') or (form_name not in valid_set):
                    if (form_name not in valid_set):
                        #print(line_name)
                        continue
                    img_path = '/'.join(pathlist)
                    
                    transcr = ' '.join(info[8:])
                    writer_name = form_writer_dict[form_name]
                    #print('writer_name', writer_name)
                    writer_name = wr_dict[writer_name]
                    
                    gt.append((img_path, transcr, writer_name))
            return gt

        info = gather_iam_info(self, subset, segmentation_level)
        data = []
        widths = []
        for i, (img_path, transcr, writer_name) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            #

            try:
                #print('img_path', img_path + '.png')
                img = Image.open(img_path + '.png').convert('RGB') #.convert('L')
                #print('img shape PIL', img.size)
                #img = image_resize_PIL(img, height=64)
                
                if img.height < 64 and img.width < 256:
                    img = img
                else:
                    img = image_resize_PIL(img, height=img.height // 2)
                
                #widths.append(img.size[0])
                
            except:
               continue
                
            #except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")
            
            data += [(img, transcr, writer_name, img_path)]
            
        return data

    
    
class Mixed_Encoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', num_classes=339, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool=""
        )
        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Create the classifier
        if hasattr(self.model, 'num_features'):
            num_features = self.model.num_features
        else:
            # Fallback, can be adjusted based on the specific model
            num_features = 2048

        self.classifier = nn.Linear(num_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        # Extract features
        features = self.model(x)

        # Pool the features to make them of fixed size
        pooled_features = self.global_pool(features).flatten(1)

        # Classify
        logits = self.classifier(pooled_features)
        # print('logits', logits.shape)
        # print('pooled_features', pooled_features.shape)
        return logits, pooled_features  


#================ Performance and Loss Function ========================
def performance(pred, label):
    
    loss = nn.CrossEntropyLoss()
   
    loss = loss(pred, label)
    return loss 

#===================== Training ==========================================

def train_class_epoch(model, training_data, optimizer, args):
    '''Epoch operation in training phase'''
    
    model.train()
    total_loss = 0
    n_corrects = 0 
    total = 0
    pbar = tqdm(training_data)
    for i, data in enumerate(pbar):
    
        image = data[0].to(args.device)
        if args.dataset == 'iam':
            label = data[2].to(args.device)
        
        optimizer.zero_grad()

        output = model(image)
        
        loss = performance(output, label)
        _, preds = torch.max(output.data, 1)
 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        total += label.size(0)
        n_corrects += (preds == label).sum().item()
        pbar.set_postfix(Loss=loss.item())
        
    loss = total_loss/total
    accuracy = n_corrects/total
    
    return loss, accuracy

def eval_class_epoch(model, validation_data, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0
    prediction_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_data)):

            image = data[0].to(args.device)   
            image_paths = data[4]
            if args.dataset == 'iam':
                label = data[2].to(args.device)

            output = model(image)
            
            loss = performance(output, label)  #performance
            _, preds = torch.max(output.data, 1)
            
            total_loss += loss.item()
            n_corrects += (preds == label.data).sum().item()
            total += label.size(0)
            #prediction_list.append(preds)
            #write into a file the img_path and the prediction
            # with open('predictions.txt', 'a') as f:
            #     for i, p in enumerate(preds):
            #         f.write(f'{image_paths[i]},{p}\n')
            
    loss = total_loss/total
    accuracy = n_corrects/total

    return loss, accuracy




########################################################################              
def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
    
        if args.dataset == 'iam':
            wid = data[2]
            #print('wid', wid)
            positive = data[3]
            negative = data[4]
        
        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(triplet_loss=loss_meter.avg)
        total += img.size(0)
    
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    return running_loss/total #np.mean(running_loss)/total

def val_epoch_triplet(val_loader, model, criterion, optimizer, device, args):
    
    running_loss = 0
    total = 0
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        #transcr = data[1]

        if args.dataset == 'iam':
            wid = data[2]
            positive = data[3]
            negative = data[4]
       
        anchor = img.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
    
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        pbar.set_postfix(triplet_loss=loss.item())
        total += wid.size(0)
    
    print('total', total)
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    return running_loss/total #np.mean(running_loss)/total



############################ MIXED TRAINING ############################################              
def train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        wid = data[3].to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img.to(device)
        # Get logits and features from the model
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        
        loss = classification_loss + triplet_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg, classification_loss=loss_meter_class.avg, triplet_loss=loss_meter_triplet.avg)
        total += img.size(0)
    
    accuracy = n_corrects/total
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    print("Training Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total

def val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0].to(device)
        wid = data[3].to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        loss = classification_loss + triplet_loss
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg)
        total += wid.size(0)
    
    print('total', total)
    accuracy = n_corrects/total
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    print("Validation Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total






#TRAINING CALLS

def train_mixed(model, train_loader, val_loader, criterion_triplet, criterion_classification, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/mixed_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
        
        
def train_classification(model, training_data, validation_data, optimizer, scheduler, device, args): #scheduler # after optimizer
    ''' Start training '''

    valid_accus = []
    num_of_no_improvement = 0
    best_acc = 0
    
    for epoch_i in range(args.epochs):
        print('[Epoch', epoch_i, ']')

        start = time.time()
        #wandb.log({'lr': scheduler.get_last_lr()})
        #print('Epoch:', epoch_i,'LR:', scheduler.get_last_lr())

        train_loss, train_acc = train_class_epoch(model, training_data, optimizer, args)
        print('Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100*train_acc,
                  elapse=(time.time()-start)/60))
        
        start = time.time()
        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}

        if validation_data is not None:
            val_loss, val_acc = eval_class_epoch(model, validation_data, args)
            print('Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                        loss=val_loss, accu=100*val_acc,
                    elapse=(time.time()-start)/60))
            
            if val_acc > best_acc:
                
                print('- [Info] The checkpoint file has been updated.')
                best_acc = val_acc
                torch.save(model.state_dict(), f"{args.save_path}/{args.dataset}_classification_{args.model}.pth")
                num_of_no_improvement = 0
            else:
                num_of_no_improvement +=1
            
        
            if num_of_no_improvement >= 10:
                        
                print("Early stopping criteria met, stopping...")
                break
        else:
            torch.save(model.state_dict(), f"{args.save_path}/{args.dataset}_classification_{args.model}.pth")

        scheduler.step()
        #wandb.log({'epoch': epoch_i, 'train loss': train_loss, 'val loss': val_loss})
        #wandb.log({'epoch': epoch_i, 'train acc': 100*train_acc, 'val acc': 100*val_acc})
        

def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_triplet(val_loader, model, criterion, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/triplet_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
        
        

def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Train Style Encoder')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--dataset', type=str, default='iam', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=320, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, required=False, help='number of training epochs')
    parser.add_argument('--pretrained', type=bool, default=False, help='use of feature extractor or not')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--save_path', type=str, default='./style_models', help='path to save models')
    parser.add_argument('--mode', type=str, default='mixed', help='mixed for DiffusionPen, triplet for DiffusionPen-triplet, or classification for DiffusionPen-triplet')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    #========= Data augmentation and normalization for training =====#
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)
    
    if args.dataset == 'iam':
    
        myDataset = IAMDataset_style
        # dataset_folder = '/usr/share/datasets_ianos'
        dataset_folder = '/path/to/iam_data/'
        aug_transforms = [lambda x: affine_transformation(x, s=.1)]
        
        train_transform = transforms.Compose([
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        
        val_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        
        #train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), transforms=train_transform)
        
        #print('len train data', len(train_data))
        #split with torch.utils.data.Subset into train and val
        validation_size = int(0.2 * len(train_data))

        # Calculate the size of the training set
        train_size = len(train_data) - validation_size

        # Use random_split to split the dataset into train and validation sets
        train_data, val_data = random_split(train_data, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
        print('len train data', len(train_data))
        print('len val data', len(val_data))
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        

        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        if val_loader is not None:
            print('Val data')
        else:
            print('No validation data')
            
        style_classes = 339
    
    else:
        print('You need to add your own dataset and define the number of style classes!!!')
    
    
    
    
    if args.model == 'mobilenetv2_100':
        print('Using mobilenetv2_100')
        model = ImageEncoder(model_name='mobilenetv2_100', num_classes=style_classes, pretrained=True, trainable=True)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        if args.pretrained == True:
            
            state_dict = torch.load(PATH, map_location=args.device)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            #print(model)
            print('Pretrained mobilenetv2_100 model loaded')
        
        
    if args.model == 'resnet18':
        print('Using resnet18')
        model = ImageEncoder(model_name=args.model, num_classes=style_classes, pretrained=True, trainable=True)
        print('Model loaded')
        #change layer to have 1 channel instead of 3
        #model.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        if args.pretrained == True:
            PATH = ''
            
            state_dict = torch.load(PATH, map_location=args.device)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            
    
    
    model = model.to(device)
    #print(model)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", patience=3, factor=0.1
    )
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    #THIS IS THE CONDITION FOR DIFFUSIONPEN
    if args.mode == 'mixed':
        criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2) 
        print('Using both classification and metric learning training')
        train_mixed(model, train_loader, val_loader, criterion_triplet, None, optimizer_ft, scheduler, device, args)
        print('finished training')
    
    
    if args.mode == 'triplet':
        train(model, train_loader, val_loader, criterion, optimizer_ft, lr_scheduler, device, args)
        print('finished training')
    
    
    elif args.mode == 'classification':
        
        train_classification(model, train_loader, val_loader, optimizer_ft, scheduler, device, args)
        print('finished training')
    
    
if __name__ == '__main__':
    main()
