from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import argparse
import os

from data_reader import FlickrDataset
from data_reader import PersonaDataset
from transformers import AutoProcessor, BlipForConditionalGeneration


def main(args):
    if args['model_name'] == 'blip':
        model_id = "Salesforce/blip-image-captioning-large"
    if args['model_name'] == 'git':
        model_id = "microsoft/git-base-coco"

    #processor = AutoProcessor.from_pretrained(model_id)
    #model = AutoModelForCausalLM.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)

    '''
    train_dataset = FlickrDataset(data_folder = args['data_folder'], processor=processor, split='train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])
    '''
    train_dataset = PersonaDataset(data_folder = args['data_folder'], processor=processor, split='train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])
    #val_dataset = PersonaDataset(data_folder = args['data_folder'], processor=processor, split='val')
    #val_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])
    #test_dataset = PersonaDataset(data_folder = args['data_folder'], processor=processor, split='test')
    #test_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])
    print ("Done loading data!!!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if args['run_type'] == 'infer':
        best_model_path = os.path.join("/data/khyathic/style/models/ckpt_ep_14.pt")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        image_path = "/data/khyathic/style/Flicker8k_Dataset/3388330419_85d72f7cda.jpg"
        image = Image.open(image_path).convert("RGB")
        text = "In a romantic way, "
        inputs = processor(image, text, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        input_ids = inputs.input_ids
        generated_ids = model.generate(pixel_values=pixel_values, input_ids = input_ids, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_caption)
        exit(1)

    model.train()

    for epoch in range(args['epochs']):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
            loss = outputs.loss
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            '''
            model.eval()
            for vidx, batch in enumerate(val_dataloader):
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)

                outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
                val_loss = outputs.loss
                break
            if val_loss < min_val_loss:
            '''
        print ("Saving model")
        torch.save(model.state_dict(),os.path.join(args['save_path'], 'ckpt_ep_'+str(epoch)+'.pt'))
            

def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', default='train', help='train or infer', required=False)
    parser.add_argument('--model_name', default='blip', help='model name', required=False)
    parser.add_argument('--debug_mode', default='false', type=str, help='debug mode', required=False)
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate', required=False)
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs', required=False)
    parser.add_argument('--batch_size', default=16, type=int, help='batch size', required=False)
    #parser.add_argument('--data_folder', default='/data/khyathic/style', type=str, help='root folder of data', required=False)
    parser.add_argument('--data_folder', default='/data/khyathic/personality-captions/', type=str, help='root folder of data', required=False)
    parser.add_argument('--save_path', default='/data/khyathic/style/models', type=str, help='root models save folder', required=False)

    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
    print ("Done!!!")
