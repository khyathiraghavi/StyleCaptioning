import torch
from PIL import Image
import pickle

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
        text = self.captions[idx].decode("utf-8").strip()
        #text = "check this out"
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding


