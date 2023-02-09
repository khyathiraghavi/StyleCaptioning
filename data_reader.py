import torch
from PIL import Image
import pickle
import codecs
import json
import os

class PersonaDataset(torch.utils.data.Dataset):
    """ Persona Captions dataset."""
    def __init__(self, data_folder, processor, split='train'):
        self.data_folder =  data_folder
        self.processor = processor
        self.split = split
        self.data_file = os.path.join(f"{data_folder}/personality_captions", f"{split}.json")
        self.image_folder_path = os.path.join(f"{data_folder}", "yfcc_images")
        with open(self.data_file, 'r') as fp:
            self.data = json.load(fp)

        self.new_data = []
        for el in self.data:
            image_path = os.path.join(self.image_folder_path, f"{el['image_hash']}.jpg")
            if not os.path.exists(image_path):
                continue
            else:
                self.new_data.append(el)

    def __len__(self):
        return len(self.new_data)

    def __getitem__(self, idx):
        dataidx = self.new_data[idx]
        image_path = os.path.join(self.image_folder_path, f"{dataidx['image_hash']}.jpg")
        image = Image.open(image_path).convert("RGB")
        personality = dataidx['personality']
        text = dataidx['comment'].strip()
        final_text = f"In a {personality} way, " + text
        encoding = self.processor(image, final_text.lower(), padding="max_length", truncation=True, return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


class FlickrDataset(torch.utils.data.Dataset):
    """Flickr 8k dataset."""

    def __init__(self, data_folder, processor, split='train'):
        self.data_folder =  data_folder
        self.processor = processor
        self.split = split
        self.file_names_humor = pickle.load(open(f"{data_folder}/FlickrStyle_v0.9/humor/train.p", 'rb'))
        self.file_names_romantic = pickle.load(open(f"{data_folder}/FlickrStyle_v0.9/romantic/train.p", 'rb'))
        
        self.captions_humor = open(f"{data_folder}/FlickrStyle_v0.9/humor/funny_train.txt", 'rb').readlines()
        self.captions_romantic = open(f"{data_folder}/FlickrStyle_v0.9/romantic/romantic_train.txt", 'rb').readlines()
        
        self.file_names_all = self.file_names_humor +  self.file_names_romantic
        self.captions = self.captions_humor + self.captions_romantic

    def __len__(self):
        return len(self.file_names_all)

    def __getitem__(self, idx):
        image_path = f"{self.data_folder}/Flicker8k_Dataset/{self.file_names_all[idx]}"

        image = Image.open(image_path).convert("RGB")
        text = self.captions[idx].decode("windows-1252").strip()
        #text = "check this out"
        if idx < len(self.captions_humor):
            final_text = "In a funny way, " + text
        else:
            final_text = "In a romantic way, " + text
        encoding = self.processor(image, final_text, padding="max_length", truncation=True, return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding





