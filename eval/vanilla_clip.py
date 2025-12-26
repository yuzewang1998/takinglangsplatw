import os

import torch
from PIL import Image
import open_clip
# need open_clip environment. conda activate open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')
#text_list = ["Neoclassical", "Georgian", "Victorian Gothic","Palladian","Edwardian Baroque"]
text_list = ["Japanese Buddhist", "Edo Period Vernacular", "Chinese Tang Dynasty","Zen Minimalist","Neoclassical"]
text = tokenizer(text_list)
counter = zero_list = [0 for _ in text_list]
image_dir = '/media/wangyz/DATA/UBUNTU_data/dataset/PT/label/temple_nara_japan'
img_list = os.listdir(image_dir)
img_list = [img for img in img_list if img.endswith('.jpg')]
print(img_list)
for img_name in img_list:
    image = preprocess(Image.open(os.path.join(image_dir,img_name))).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        index = torch.argmax(text_probs,dim=1)
        counter[index]+=1


print(text_list)
print(counter)


image = preprocess(Image.open("/media/wangyz/DATA/UBUNTU_data/dataset/PT/label/trevi_fountain/00234320_6263685886.jpg")).unsqueeze(0)


