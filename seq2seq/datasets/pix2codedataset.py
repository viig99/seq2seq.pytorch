from __future__ import print_function
import os
import cv2
import re

class Pix2CodeDataset:
    def __init__(self, path, image_size=(256,256)):
        self.file_names = []
        self.image_size = image_size
        self.load_dataset(path=path)

    def load_dataset(self, path):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".yaml.nn") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".yaml.nn")]
                self.add_filenames("{}/{}".format(path, file_name))
        print("Dataset Loaded...")

    def get_sample(self, idx):
        return self.get_preprocessed_img(idx), self.get_preprocessed_tokens(idx)

    def get_preprocessed_img(self, idx):
        img_path = self.file_names[idx] + ".png"
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.image_size)
        return img

    def get_preprocessed_tokens(self, idx):
        gui = open(self.file_names[idx] + ".yaml.nn", 'r')
        token_sequence = []
        for line in gui:
            line = line.replace("  ", " \\t ").replace("=", " = ").replace("\"", " \" ").replace("\n", " \\n ")
            line = re.sub( '\s+', ' ', line).strip()
            tokens = line.split(" ")
            for token in tokens:
                token_sequence.append(token)
        return token_sequence

    def add_filenames(self, sample_id):
        self.file_names.append(sample_id)

    def __len__(self):
        return len(self.file_names)