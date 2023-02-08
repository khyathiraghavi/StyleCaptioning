from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

from data_reader import FlickrDataset
from transformers import AutoProcessor, BlipForConditionalGeneration


def main(args):
    if args['model_name'] == 'git':
        #model_id = "microsoft/git-base-coco"
        model_id = "Salesforce/blip-image-captioning-base"

    #processor = AutoProcessor.from_pretrained(model_id)
    #model = AutoModelForCausalLM.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    train_dataset = FlickrDataset(data_folder = args['data_folder'], processor=processor, split='train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

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
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    '''

def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='git', help='model name', required=False)
    parser.add_argument('--debug_mode', default='false', type=str, help='debug mode', required=False)
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate', required=False)
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs', required=False)
    parser.add_argument('--batch_size', default=16, type=int, help='batch size', required=False)
    parser.add_argument('--data_folder', default='/data/khyathic/style', type=str, help='root folder of data', required=False)

    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
