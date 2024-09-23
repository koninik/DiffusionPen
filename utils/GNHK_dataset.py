import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from PIL import Image, ImageOps
import json
import os
import string

class GNHK_Dataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size,  tokenizer, text_encoder, feat_extractor, transforms, args):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args)
        self.setname = 'GNHK'
        self.trainset_file = f'{self.basefolder}/GNHK_words_train.txt'
        self.testset_file = f'{self.basefolder}/GNHK_words_test.txt'
        self.word_path = self.basefolder
        
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        
        
        def gather_iam_info(self, set='train'):
            gtfile = self.trainset_file if subset == 'train' else self.testset_file
            gt = []
            folder = 'train_words' if subset == 'train' else 'test_words'
            for line in open(gtfile):
                if line.strip():
                    image_name, transcription, style = line.strip().split(' ')
                    img_path = os.path.join(self.word_path, folder, image_name)
                    
                    gt.append((img_path, transcription, style))
            return gt

        info = gather_iam_info(self, subset)
        data = []
        widths = []
        wr_dict = {}
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
        for i, (img_path, transcr, writer_name) in enumerate(info):
            
            #create writer indexes 
            if writer_name not in wr_dict:
                wr_dict[writer_name] = len(wr_dict)
                
            style = wr_dict[writer_name]
            
            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            # for cc in special_cases:
            #     transcr = transcr.replace("|\'" + cc, "\'" + cc)
            #     transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            # transcr = transcr.replace("|", " ")
            
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
              
            try:
                
                img_original = Image.open(img_path).convert('RGB') #.convert('L')
                
                #if the transcription is in stopwords
                if transcr in string.punctuation:
                    img = centered_PIL(img_original, (64, 256), border_value=255.0)
                
                else:
                    (img_width, img_height) = img_original.size
                    #resize image to height 64 keeping aspect ratio
                    img = img_original.resize((int(img_width * 64 / img_height), 64))
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
                
            except:
               continue
            
            
            
            
            
            data += [(img, transcr, style, img_path)]
            
        print('len data', len(data))
        
        #save writer_dict
        with open(f'writer_dict_train_gnhk.json', 'w') as f:
            json.dump(wr_dict, f)
        
        return data