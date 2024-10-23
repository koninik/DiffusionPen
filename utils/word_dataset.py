import io,os
import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from utils.auxilary_functions import image_resize_PIL, centered_PIL
import tqdm
from torchvision.utils import save_image
import json
import random
#import sys
#import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

#OV = True
MAX_CHARS = 40
OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64


def labelDictionary():
    #labels = list(string.ascii_lowercase + string.ascii_uppercase)
    #print('labels',labels)
    labels = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    letter2index = {label: n for n, label in enumerate(labels)}
    # create json object from dictionary if you want to save writer ids
    json_dict_l = json.dumps(letter2index)
    l = open("letter2index.json","w")
    l.write(json_dict_l)
    l.close()
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    l = open("index2letter.json","w")
    l.write(json_dict_i)
    l.close()
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 80}
else:
    tokens = {"GO_TOKEN": 80, "END_TOKEN": 81, "PAD_TOKEN": 82}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)

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
        tokenizer = None,
        text_encoder = None,
        feat_extractor = None,
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
                                #Feature extractor to be used for text encoding
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
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.output_max_len = OUTPUT_MAX_LEN
        self.feat_extractor = feat_extractor
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
        
        save_path = './saved_iam_data'
        
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        save_file = '{}/{}_{}_{}.pt'.format(save_path, self.subset, self.segmentation_level, self.setname) #dataset_path + '/' + set + '_' + level + '_IAM.pt'
        
        if isfile(save_file) is False:
            data = self.main_loader(self.subset, self.segmentation_level)
            torch.save(data, save_file)   #Uncomment this in 'release' version
        else:
            data = torch.load(save_file)
        
        #data = self.main_loader(self.subset, self.segmentation_level)
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
            for _,transcr,_,_ in tqdm.tqdm(data):
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

    @staticmethod
    def draw_word(word: str) -> Image:
        # Define the target image width and height
        target_width = 256
        target_height = 64

        # Calculate the appropriate font size based on the target width and word length
        max_font_size = 45
        text_width, text_height = float('inf'), float('inf')
        font_size = max_font_size
        while text_width > target_width or text_height > target_height:
            font_size -= 1
            font = ImageFont.truetype('./Roboto-Regular.ttf', font_size)
            _,_,text_width, text_height = font.getbbox(word)
            
        # Create a white image with the target dimensions
        img = Image.new('RGB', (target_width, target_height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # Calculate the position to center the text
        position = ((target_width - text_width) / 2, (target_height - text_height) / 2)

        # Draw the text onto the image
        d.text(position, word, font=font, fill=0)

        return img

    @staticmethod
    def find_text_bounding_box(image):
            # Load the image
        #image = cv2.imread(image_path)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Threshold the image to separate black text from the background
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("Number of contours detected:", len(contours))
        cnts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(cnts)
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
        #cv2.imwrite('./new.png', image)
        
        return (x, y, w, h)
    
    @staticmethod
    def draw_word_in_bounding_box(word: str, bounding_box: tuple) -> Image:
        # bounding_box is a tuple (x1, y1, x2, y2) specifying the top-left (x1, y1) and
        # bottom-right (x2, y2) coordinates of the bounding box

        # Create a white image with the target dimensions (64x256)
        target_width = 256
        target_height = 64
        img = Image.new('RGB', (target_width, target_height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # Calculate the width and height of the bounding box
        box_width = bounding_box[2] - bounding_box[0]
        box_height = bounding_box[3] - bounding_box[1]

        # Calculate the appropriate font size based on the bounding box dimensions and word length
        max_font_size = 50
        font_size = max_font_size

        while True:
            # Load the font
            font = ImageFont.truetype('./Roboto-Regular.ttf', font_size)

            # Get the size of the text with the current font
            text_width, text_height = d.textsize(word, font=font)

            # Check if the text fits within the bounding box
            if text_width <= box_width and text_height <= box_height:
                break  # The text fits, exit the loop
            else:
                font_size -= 1  # Reduce font size and try again

        # Calculate the position to center the text within the bounding box
        x = bounding_box[0] + (box_width - text_width) / 2
        y = bounding_box[1] + (box_height - text_height) / 2
        position = (x, y)

        # Draw the text onto the image
        d.text(position, word, font=font, fill=0)

        return img
    
    def __getitem__(self, index):
        
        img = self.data[index][0]
        img_path = self.data[index][3]
        if self.transforms is not None:
            img = self.transforms(img)
        #save_image(img, 'check_style.png')
        transcr = self.data[index][1]
        wid = self.data[index][2]
 
        #pick another sample that has the same self.data[2] or same writer id
        positive_samples = [p for p in self.data if p[2] == wid and len(p[1])>3]
        #negative_samples = [p for p in self.data if p[2] != wid and len(p[1])>3]
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
            random_samples_ = random.sample(positive_samples_, k=5)
            # Retrieve the corresponding images
            style_images = [i[0] for i in random_samples_]
        
        cor_images = random.sample(positive_samples, k=1)
        cor_im = cor_images[0][0]
        cor_im = self.transforms(cor_im)
        
        '''
        pos_image = random.sample(positive_samples, k=1)
        neg_image = random.sample(negative_samples, k=1)
        
        pos_image = pos_image[0][0]
        neg_image = neg_image[0][0]
        pos_image = self.transforms(pos_image)
        neg_image = self.transforms(neg_image)
        '''
        st_imgs = []
        for s_img in style_images:
            
            if self.transforms is not None:
                s_img_tensor = self.transforms(s_img)
            
            st_imgs += [s_img_tensor]
            
        s_imgs = torch.stack(st_imgs)
    
        return img, transcr, wid, s_imgs, img_path, cor_im#, pos_image, neg_image #, style_features#, printed_word, bbox#, tok_transcr
    
    def collate_fn(self, batch):
        # Separate image tensors and caption tensors
        img, transcr, wid, s_imgs, img_path, cor_im = zip(*batch)

        #context = [item.detach() for item in transcr]  # Detach context tensors
        transcr = torch.stack(transcr)
        #context = tok_transcr#torch.stack(tok_transcr)
        
        # Stack image tensors and caption tensors into batches
        images_batch = torch.stack(img)
        
        s_imgs = torch.stack(s_imgs)
        
        style_features = torch.stack(style_features)
        #printed_word = torch.stack(printed_word)
        #bbox = torch.stack(bbox)
        cor_images_batch = torch.stack(cor_im)
        # pos_images_batch = torch.stack(pos_image)
        # neg_images_batch = torch.stack(neg_image)
        
        return images_batch, transcr, wid, s_imgs, img_path, cor_images_batch#, pos_images_batch, neg_images_batch#, printed_word, bbox#, context

    
    
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

