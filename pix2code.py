#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.utils.model_zoo import load_url
from seq2seq.models import Img2Seq
from seq2seq.tools.inference import CaptionGenerator
from seq2seq.tools.bleu import get_bleu
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.misc import imresize
import cv2
from seq2seq.datasets.pix2codedataset import Pix2CodeDataset
import progressbar

def show_and_tell(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256,256))
    plt.figure()
    plt.imshow(np.asarray(img))
    return img, caption_model.describe(img)

def sample(dset, caption_model, index=None):
    sample_index = np.random.choice(len(dset)) if index is None else index
    sample_img_filename = dset.file_names[sample_index] + ".png"
    sample_img, sample_target = dset.get_sample(sample_index)
    predicted_target = caption_model.describe(sample_img)
    return (sample_img, sample_img_filename), sample_target, predicted_target.decode('utf-8').split(' ')

def return_list_with_tabs(li):
    elems = map(lambda x: x.replace("\\n","\n").replace("\\t", "\t"), li)
    return ' '.join(elems)

def get_bleu_score(dset, caption_model, num=1):
    refs = []
    hyps = []
    bar = progressbar.ProgressBar()
    for i in bar(range(num)):
        _, ref, hyp = sample(dset, caption_model, i)
        refs.append(ref)
        hyps.append(hyp)
    return get_bleu(hyps, refs)

def print_a_sample(dset, caption_model):
    (img, img_fname), target, predicted = sample(dset, caption_model)
    print("Sample Image Path: \n", img_fname)
    print("Sample Target: \n", return_list_with_tabs(target))
    print("Sample Predicted: \n", return_list_with_tabs(predicted))

if __name__ == '__main__':
    checkpoint = load_url(
        'model_best.pth.tar', model_dir="results/pix2code_devsupport_densenet121_finetune/", map_location={'gpu:0': 'cpu'})
    model = Img2Seq(**checkpoint['config'].model_config)
    model.load_state_dict(checkpoint['state_dict'])
    img_transform, target_tok = checkpoint['tokenizers'].values()
    beam_size = 3
    caption_model = CaptionGenerator(model,
                                     img_transform=img_transform,
                                     target_tok=target_tok,
                                     beam_size=beam_size,
                                     get_attention=False,
                                     max_sequence_length=300,
                                     length_normalization_factor=2.0,
                                     cuda=True,
                                     length_normalization_const=2)

    dset = Pix2CodeDataset("/home/vigi99/devsupportai_ui_gen/eval_set/")
    # print_a_sample(dset, caption_model)
    num_of_examples = len(dset)
    bleu = get_bleu_score(dset, caption_model, num_of_examples)
    print('Bleu Score for {0:d} examples with beam size {2:d} is {1:.2f}'.format(num_of_examples, bleu, beam_size))