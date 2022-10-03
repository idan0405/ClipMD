from transformers import CLIPProcessor,CLIPModel,CLIPConfig
from functions import *
import glob
torch.cuda.empty_cache()

images_path=input("Write the path to the folder that includes your images in .png files: ")
texts_path=input("Write the path to the .txt folder that includes your text descriptions (separated by new lines): ")
device=torch.device(input("Write the device you want torch to use (cpu/cuda): "))

config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model.to(device)
model.load_state_dict(torch.load("clipmd.pt"))

images=[]
for f in glob.glob(f"{images_path}/*.png"):
    images.append(Image.open(f))


with open("./"+texts_path, "r") as f:
    texts=f.read().splitlines()

probs = calc_logits(texts, images, device, model, config, processor)[0].softmax(dim=1)
print(probs)
