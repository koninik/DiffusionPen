import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from PIL import Image, ImageOps
import json
import os
import string

class IAMDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size,  tokenizer, text_encoder, feat_extractor, transforms, args):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args)
        self.setname = 'IAM'
        self.trainset_file = '{}/{}/set_split/trainset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/set_split/validationset1.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/set_split/testset.txt'.format(self.basefolder, self.setname)
        self.line_file = '{}/ascii/lines.txt'.format(self.basefolder, self.setname)
        self.word_file = './iam_data/ascii/words.txt'.format(self.basefolder, self.setname)
        #self.word_path = '{}/words'.format(self.basefolder, self.setname)
        self.word_path = self.basefolder
        self.line_path = '{}/lines'.format(self.basefolder, self.setname)
        self.forms = './iam_data/ascii/forms.txt'
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feat_extractor = feat_extractor
        self.args = args
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def generate_multiple_crops(img, num_crops=4, crop_size=(200, 50)):
        crops = []
        for _ in range(num_crops):
            max_x = img.size[0] - crop_size[0]
            max_y = img.size[1] - crop_size[1]
            if max_x <= 0 or max_y <= 0:  # Ensuring the crop size is smaller than the image
                # If the image is too small to be cropped, resize the original image instead
                resized_img = img.resize((crop_size[0], crop_size[1]))
                crops.append(resized_img)
            else:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                crop = img.crop((x, y, x + crop_size[0], y + crop_size[1]))
                crops.append(crop)
        return crops
    
    
    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(self, set='train', level='word'):
            if subset == 'train':
                #valid_set = np.loadtxt(self.trainset_file, dtype=str)
                valid_set = np.loadtxt('./aachen_iam_split/train_val.uttlist', dtype=str)
                #print(valid_set)
            elif subset == 'val':
                #valid_set = np.loadtxt(self.valset_file, dtype=str)
                valid_set = np.loadtxt('./aachen_iam_split/validation.uttlist', dtype=str)
            elif subset == 'test':
                #valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt('./aachen_iam_split/test.uttlist', dtype=str)
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
        padded_imgs = 0
        padded_data = []
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        for i, (img_path, transcr, writer_name) in enumerate(info):
            
            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")
            
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
              
            try:
                
                img = Image.open(img_path + '.png').convert('RGB') #.convert('L')
                
                #if the transcription is in stopwords
                if transcr in string.punctuation:
                    img = centered_PIL(img, (64, 256), border_value=255.0)
                
                else:
                    (img_width, img_height) = img.size
                    #resize image to height 64 keeping aspect ratio
                    img = img.resize((int(img_width * 64 / img_height), 64))
                    (img_width, img_height) = img.size
                    
                    if img_width < 256:
                        outImg = ImageOps.pad(img, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                        img = outImg
                    
                    else:
                        #reduce image until width is smaller than 256
                        while img_width > 256:
                            img = image_resize_PIL(img, width=img_width-20)
                            (img_width, img_height) = img.size
                        img = centered_PIL(img, (64, 256), border_value=255.0)
                        #img = image_resize_PIL(img, height=img.height // 2)
                
                #img.save(f'/home/konnik/AFINAL_CVPR/check_padding/img_new_padding_{i}.png')
                '''
                img_padded = False
                if subset == 'train' and writer_name!=12:
                    # Create a new image by concatenating the original image with itself
                    padded_image = Image.new('RGB', (img.width * 2, img.height))
                    padded_image.paste(img, (0, 0))
                    padded_image.paste(img, (img.width, 0))
                    #padded_image.save('selfpadded.png')
                    # Construct the new image path with "_padded" added to the filename 
                    img_padded = True
                    wr_padded = writer_name
                    transcr_pad = transcr*2
                    padded_imgs += 1
                '''
            except:
               continue
            
            
            
            
            data += [(img, transcr, writer_name, img_path)]
            '''
            if img_padded:
                
                padded_data.append((padded_image, transcr_pad, wr_padded, img_path))
            
            #padded_data += [(padded_image, transcr*2, writer_name, img_path)]
            img_padded = False
            '''  
            #with open('/home/konnik/iam_data/splits_words/iam_test.txt', 'a') as f:
             #   path = img_path.split('/')[-3:]
              #  im_path = '/'.join(path) + '.png'
               # print('img_path', im_path)
                #f.write('{},{},{}\n'.format(im_path, writer_name, transcr))
        #print('max line width', max(widths))
        #print('len widths', len(widths))
        print('len data', len(data))
        #print('len padded', len(padded_data))
        
        #merge data and padded_data
        data_full = data #+ padded_data
        print('len data_full', len(data_full))
        
        return data_full
