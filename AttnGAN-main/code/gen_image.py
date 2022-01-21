#importing libraries

from __future__ import print_function

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

import numpy as np
from torch.autograd import Variable
from PIL import Image

#different classes (or self defined dunctions)
from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from miscc.utils import build_super_images2

import os

#functions important for generating images from text
#####Tokenize sentence
def tokenize_sent(sent,wordtoix):
  from nltk.tokenize import RegexpTokenizer
  if len(sent) == 0:
    return "Sentence was empty"
  sent = sent.replace("\ufffd\ufffd", " ")
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(sent.lower())
  if len(tokens) == 0:
    print('sent', sent)

  rev = []
  for t in tokens:
    t = t.encode('ascii', 'ignore').decode('ascii')
    if len(t) > 0 and t in wordtoix:
        rev.append(wordtoix[t])
  return [rev]

#####function which will generate images from given sentence
def generate_image_sent(sent,model_values):
  algo,text_encoder,netG,dataset=model_values
  my_caption=tokenize_sent(sent,dataset.wordtoix)
  my_cap_len=[len(my_caption[0])]
  
  #converting things into their proper forms
  batch_size = 1
  nz = cfg.GAN.Z_DIM
  my_caption = Variable(torch.from_numpy(np.array(my_caption)), volatile=True)
  my_cap_len = Variable(torch.from_numpy(np.array(my_cap_len)), volatile=True)

  my_caption = my_caption.type(torch.LongTensor)##changed this, fuck this line really

  if cfg.CUDA:
    my_caption = my_caption.cuda()
    my_cap_len = my_cap_len.cuda()


  #generating noise, mask and impt embeddings

  noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
  if cfg.CUDA:
    noise = noise.cuda()
  #######################################################
  # (1) Extract text embeddings
  ######################################################
  hidden = text_encoder.init_hidden(batch_size)
  # words_embs: batch_size x nef x seq_len
  # sent_emb: batch_size x nef
  words_embs, sent_emb = text_encoder(my_caption, my_cap_len, hidden)
  mask = (my_caption == 0)
  #######################################################
  # (2) Generate fake images
  ######################################################
  noise.data.normal_(0, 1)
  #print(noise, sent_emb, words_embs, mask)


  #Generating (Fake)Images
  my_fake_imgs, my_attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

  #important for extracting text back from tokenized form
  my_cap_lens_np = my_cap_len.cpu().data.numpy()

  #saving images
  for j in range(batch_size):     #which is always 1 for  sentance will remove this loop soon

    #save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
    save_name='output/my_img' #name any folder, right now its named output which you have to create manually inside AttnGAN/code

    for k in range(len(my_fake_imgs)):
        im = my_fake_imgs[k][j].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        # print('im', im.shape)
        im = np.transpose(im, (1, 2, 0))
        # print('im', im.shape)
        im = Image.fromarray(im)
        fullpath = '%s_g%d.png' % (save_name, k)
        im.save(fullpath)

    for k in range(len(my_attention_maps)):
        if len(my_fake_imgs) > 1:
            im = my_fake_imgs[k + 1].detach().cpu()
        else:
            im = my_fake_imgs[0].detach().cpu()
        attn_maps = my_attention_maps[k]
        att_sze = attn_maps.size(2)
        img_set, sentences = \
            build_super_images2(im[j].unsqueeze(0),
                                my_caption[j].unsqueeze(0),
                                [my_cap_lens_np[j]], algo.ixtoword,
                                [attn_maps[j]], att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s_a%d.png' % (save_name, k)
            im.save(fullpath)





#setting current directory
current_dir=os.getcwd()

def loading_model(dataset_name='bird'):
  #IMPORTANT ARGUMENTS
  if (dataset_name=='bird') :
    cfg_file=os.path.join(current_dir,"cfg/eval_bird.yml")
  else :
    cfg_file=os.path.join(current_dir,"cfg/eval_coco.yml")
  
  gpu_id=-1 #change it to 0 or more when using gpu
  data_dir=''
  manualSeed = 100

  #cfg file set
  if cfg_file is not None:
    cfg_from_file(cfg_file)

  if gpu_id != -1:
    cfg.GPU_ID = gpu_id
  else:
    cfg.CUDA = False

  if data_dir != '':
    cfg.DATA_DIR = data_dir


  now = datetime.datetime.now(dateutil.tz.tzlocal())
  timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
  output_dir = '../output/%s_%s_%s' % \
    (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

  split_dir, bshuffle = 'train', True
  if not cfg.TRAIN.FLAG:
    # bshuffle = False
    split_dir = 'test'


  # Get data loader
  imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
  image_transform = transforms.Compose([
      transforms.Scale(int(imsize * 76 / 64)),
      transforms.RandomCrop(imsize),
      transforms.RandomHorizontalFlip()])
  dataset = TextDataset(cfg.DATA_DIR, split_dir,
                        base_size=cfg.TREE.BASE_SIZE,
                        transform=image_transform)
  assert dataset
  dataloader = torch.utils.data.DataLoader(
          dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
          drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))


  ###setting up ALGO
  # Define models and go to train/evaluate
  algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

  #loading text ENCODER
  text_encoder = RNN_ENCODER(algo.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
  state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage) #TRAIN.NET_E path can be given directly
  text_encoder.load_state_dict(state_dict)
  # print('Load text encoder from:', cfg.TRAIN.NET_E) ###edited here
  if cfg.CUDA:
    text_encoder = text_encoder.cuda()
  text_encoder.eval()


  #LOADING Generator
  netG = G_NET()
  model_dir = cfg.TRAIN.NET_G #directory for model can be given directly as well
  state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
  netG.load_state_dict(state_dict)
  # print('Load G from: ', model_dir)  ###edited here
  if cfg.CUDA:
    netG.cuda()
  netG.eval()

  return [algo,text_encoder,netG,dataset]


#finally running the main function
if __name__ == '__main__':
  ans='y'
  # coco_model=loading_model('coco')
  bird_model=loading_model('bird')
  while(ans=='y' or ans=='Y'):
    my_sent=input("Enter description : ")
    generate_image_sent(my_sent,bird_model)
    ans=input("Wanna generate again? (Y/N): ")