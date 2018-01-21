import os
import logging
from random import randrange
from collections import OrderedDict
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
from PIL import ImageFile
from seq2seq.datasets.pix2codedataset import Pix2CodeDataset
from seq2seq.tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer
from seq2seq.tools import batch_sequences
from seq2seq.tools.config import EOS, BOS, PAD, LANGUAGE_TOKENS

def imagenet_transform():
    normalize = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((620,350)),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])

def create_padded_tokens_batch(max_length=None, max_tokens=None, batch_first=False,
                                sort=True, pack=True, augment=False):
    def collate(img_seq_tuple):
        if sort or pack:  # packing requires a sorted batch by length
            img_seq_tuple = sorted(img_seq_tuple, key=lambda p: len(p[1]), reverse=True)
        imgs, seqs = zip(*img_seq_tuple)
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        seq_tensor = batch_sequences(seqs, max_length=max_length,
                                     max_tokens=max_tokens,
                                     batch_first=batch_first,
                                     sort=sort, pack=pack, augment=augment)
        return (imgs, seq_tensor)
    return collate


class Pix2Code(Dataset):
    """docstring for Dataset."""

    def __init__(self, root, img_transform=imagenet_transform,
                 split='train',
                 vocab_file=None,
                 insert_start=[BOS], 
                 insert_end=[EOS],
                 tokenizer=None):
        super(Pix2Code, self).__init__()
        self.tokenizer = tokenizer
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.vocab_file = vocab_file
        self.img_transform = img_transform()
        path = "data" if split == "train" else "eval_set"
        self.dset = Pix2CodeDataset(path=os.path.join(root, path))

        if self.tokenizer is None:
            prefix = os.path.join(root, 'pix2code_devsupport')
            self.vocab_file = vocab_file or '{prefix}.{lang}.{tok}.vocab{num_symbols}'.format(
                prefix=prefix, lang='en', tok='word', num_symbols='')
            self.generate_tokenizer()

    def generate_tokenizer(self):
        tokz = Tokenizer(
            vocab_file=self.vocab_file,
            additional_tokens=None)

        if not hasattr(tokz, 'vocab'):
            sentences = [self.dset.get_preprocessed_tokens(i) for i in range(len(self.dset))]
            logging.info('generating vocabulary. saving to %s' %
                         self.vocab_file)
            tokz.get_vocab(sentences, from_filenames=False, segmented=True)
            tokz.save_vocab(self.vocab_file)
        self.tokenizer = tokz

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]
        img, captions = self.dset.get_sample(index)
        insert_start = self.insert_start
        insert_end = self.insert_end

        def transform(t):
            return self.tokenizer.tokenize(t,
                                           insert_start=insert_start,
                                           insert_end=insert_end,
                                           segmented=True)
        img = self.img_transform(img)
        captions = transform(captions)
        return (img, captions)

    def __len__(self):
        return len(self.dset)

    def get_loader(self, batch_size=32, shuffle=True, pack=True, sampler=None, num_workers=0,
                   max_length=None, max_tokens=None, batch_first=False,
                   pin_memory=False, drop_last=False, augment=False):
        collate_fn = create_padded_tokens_batch(
            max_length=None, max_tokens=None,
            pack=pack, batch_first=batch_first, augment=augment)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           collate_fn=collate_fn,
                                           sampler=sampler,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last)

    @property
    def tokenizers(self):
        return OrderedDict(img=self.img_transform, en=self.tokenizer)
