from torch import optim
from ClipMDModel import *
from transformers import CLIPProcessor
from split_roco import *
from PIL import Image
import math
import random
import sys
BATCH_SIZE=50
NUM_OF_EPOCHS=10

def train(path):
    torch.cuda.empty_cache()
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #loading in the preprocessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #loading in the training data
    raw_train_data=get_train_data(path)

    train_set=[]
    for entry in raw_train_data:
        try:
            Image.open(entry[0])
            train_set.append(entry)
        except:
            continue

    #loading in the validation data
    raw_val_data=get_validation_data(path)

    val_set=[]
    for entry in raw_val_data:
        try:
            Image.open(entry[0])
            val_set.append(entry)
        except:
            continue

    #loading in the model and seting up the optimizer
    model = ClipMDModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6)

    #fine-tuning the model
    for epoch in range(1,NUM_OF_EPOCHS+1):

        #shuffleing the data
        random.shuffle(train_set)

        total_loss=0
        for i in range(0,len(train_set),BATCH_SIZE):

            #creating a batch of size BATCH_SIZE
            batch=train_set[i:i+BATCH_SIZE]
            batch = list(zip(*batch))
            imgs,texts=list(batch[0]),list(batch[1])

            #opening the images
            ims=[]
            for im in imgs:
                ims.append(Image.open(im))

            #calculating the loss
            inputs=processor(text=texts, images=ims, return_tensors="pt", padding=True)
            loss = model(input_ids = inputs["input_ids"].to(device),
            pixel_values = inputs["pixel_values"].to(device),
            attention_mask = inputs["attention_mask"].to(device),
                         return_loss=True)[-1]
            total_loss+=loss.item()

            #backpropagating the loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        #saving the weights
        torch.save(model.state_dict(), "clipmd-roco2.pt")

        with torch.no_grad():

            #running the model on the validation set
            val_loss=0
            for i in range(0,len(val_set),BATCH_SIZE):

                #creating a batch of size BATCH_SIZE
                batch=val_set[i:i+BATCH_SIZE]
                batch = list(zip(*batch))
                imgs, texts = list(batch[0]), list(batch[1])

                #opening the images
                ims = []
                for im in imgs:
                    ims.append(Image.open(im))

                #calculating the loss
                inputs=processor(text=texts, images=ims, return_tensors="pt", padding=True)
                loss = model(input_ids = inputs["input_ids"].to(device),
                pixel_values = inputs["pixel_values"].to(device),
                attention_mask = inputs["attention_mask"].to(device),
                             return_loss=True)[-1]
                val_loss+=loss.item()
                torch.cuda.empty_cache()

        print(f"epoch: {epoch}, loss:{total_loss/math.ceil(len(train_set)/BATCH_SIZE):.4f},"
              f"val_loss:{val_loss/math.ceil(len(val_set)/BATCH_SIZE):.4f}")

if __name__ == "__main__":
    train(sys.argv[1])
