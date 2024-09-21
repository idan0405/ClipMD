from ClipMDModel import *
from transformers import CLIPProcessor
from torchmetrics import Recall
from PIL import Image
import random
from split_roco import *
import sys
BATCH_SIZE = 512

def recall_at_k(path):
    #loading the test set
    raw_test_data=get_test_data(path)

    data_list=[]
    for entry in raw_test_data:
        try:
            data_list.append([Image.open(entry[0]),entry[1]])
        except:
            continue
    #randomly sampling 2000 image/caption pairs
    data_list=random.sample(data_list,2000)
    data_list=list(zip(*data_list))
    with torch.no_grad():
        torch.cuda.empty_cache()

        device = "cuda"

        #loading the preprocessor and the model
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = ClipMDModel.from_pretrained("openai/clip-vit-base-patch32")

        #loading the model onto the device and laoding in the weights
        model.to(device)
        model.load_state_dict(torch.load("clipmd-roco.pt"))

        #spliting the data into images and texts
        images,texts = list(data_list[0]),list(data_list[1])

        #producing the logits matrix of the images and the texts
        logits=[]
        for i in range(0,len(images),BATCH_SIZE):
            ims = images[i:i+BATCH_SIZE]

            logit=None
            for j in range(0,len(texts),BATCH_SIZE):
                inputs=processor(text=texts[j:j+BATCH_SIZE], images=ims, return_tensors="pt", padding=True)
                new_logit=model(input_ids = inputs["input_ids"].to(device),
                                pixel_values = inputs["pixel_values"].to(device),
                                attention_mask = inputs["attention_mask"].to(device)
                                )[0].detach().to("cpu")

                if logit is not None:
                    logit=torch.cat((logit,new_logit),1)
                else:
                    logit=new_logit

            logits.append(logit)

        logits=torch.cat(logits,0)

        #printing the Recall at K for logits matix (k=1, 5, 10, 20)
        print(Recall(task="multiclass", num_classes=2000, top_k=1)(logits, torch.arange(2000)))
        print(Recall(task="multiclass", num_classes=2000, top_k=5)(logits, torch.arange(2000)))
        print(Recall(task="multiclass", num_classes=2000, top_k=10)(logits, torch.arange(2000)))
        print(Recall(task="multiclass", num_classes=2000, top_k=20)(logits, torch.arange(2000)))

if __name__ == "__main__":
    recall_at_k(sys.argv[1])
