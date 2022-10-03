import torch
from torch import nn
from PIL import Image


def encode_text(inputs,device,model):
    """
    :param inputs: CLIPProcessor's output.
    :param device: the device that all of the tensors should be on.
    :param model: an instance of CLIPModel.
    :return: text encodings of inputs['input_ids'] (texts longer then 77 tokens
    is encoded using a sliding window and pooling.
    """
    texts_encodings=[]
    texts = []
    masks=[]
    pos=[]
    for i in range(len(inputs['input_ids'])):
        ten=inputs['input_ids'][i]
        mask=inputs['attention_mask'][i]

        mask = torch.transpose(mask[mask.nonzero()], 0, 1)
        if ten.size()[0]>77:
            ten = ten[:max(mask.size()[1],77)].unsqueeze(0)
        else:
            ten = ten[:mask.size()[1]].unsqueeze(0)

        if ten.size()[1]<=77:

            zero=torch.zeros((1,max(ten.size()[1],77))).to(device)
            zero[:,0:mask.size()[1]]=mask[0]

            new_ten=torch.full((1,max(zero.size()[1],77)), 49407).to(device)
            new_ten[:,0:ten.size()[1]]=ten[0]

            mask=zero
            ten=new_ten

            if not pos:
                pos.append([0,1])
            else:
                pos.append([pos[-1][1],pos[-1][1]+1])

            texts.append(ten)
            masks.append(mask)

        else:

            if not pos:
                pos.append([0, 0])
            else:
                pos.append([pos[-1][1], pos[-1][1]])

            for i in range(0,ten.size()[1],70):
                pos[-1][1]+=1
                if i+77>=ten.size()[1]:
                    zero = torch.zeros((1,  77)).to(device)
                    zero[:, 0:mask.size()[1]-i] = mask[:,i:mask.size()[1]]

                    new_ten = torch.full((1,  77), 49407).to(device)
                    new_ten[:, 0:ten.size()[1]-i] = ten[:,i:ten.size()[1]]

                    #texts.append(ten[:, ten.size()[1] - 77:ten.size()[1]])
                    #masks.append(mask[:,ten.size()[1]-77:ten.size()[1]])
                    texts.append(new_ten)
                    masks.append(zero)
                else:
                    texts.append(ten[:, i:i + 77])
                    masks.append(mask[:,i :i+77])

    encoded=model.get_text_features(input_ids=torch.cat(texts,0),attention_mask=torch.cat(masks,0))

    for p in pos:
        if p[1]-p[0]==1:
            texts_encodings.append(encoded[p[0]].unsqueeze(0))
        else:
            texts_encodings.append(torch.mean(encoded[p[0]:p[1]],dim=0).unsqueeze(0))

    return torch.cat(texts_encodings,0)


def calc_logits(texts, imgs, device,model,config,processor):
    """
    :param texts: list of text descriptions.
    :param imgs: list of images.
    :param device: the device that all of the tensors should be on.
    :param model: instance of CLIPModel.
    :param config: instance of CLIPConfig.
    :param processor: instance of CLIPProcessor.
    :return: the logits per image and per text.
    """
    inputs = processor(text=list(texts), images=imgs, return_tensors="pt", padding=True)

    inputs['input_ids'] = inputs['input_ids'].long().to(device)
    inputs['attention_mask'] = inputs['attention_mask'].float().to(device)
    inputs['pixel_values'] = inputs['pixel_values'].float().to(device)

    texts_encodings=encode_text(inputs,device,model)
    images_encodings=model.get_image_features(inputs['pixel_values'])

    images_encodings = images_encodings / images_encodings.norm(p=2, dim=-1, keepdim=True)
    texts_encodings = texts_encodings / texts_encodings.norm(p=2, dim=-1, keepdim=True)

    logit_scale=nn.Parameter(torch.ones([]) * config.logit_scale_init_value)
    logit_scale = logit_scale.exp()

    logits_per_text = torch.matmul(texts_encodings, images_encodings.t()) * logit_scale
    logits_per_image = logits_per_text.T

    return logits_per_image, logits_per_text


def contrastive_loss(logits: torch.Tensor):
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor):
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0

def calc_loss(batch,device,model,config,processor):
    """
    :param batch: list of lists of texts and image paths.
    :param device: the device that all of the tensors should be on.
    :param model: instance of CLIPModel.
    :param config: instance of CLIPConfig.
    :param processor: instance of CLIPProcessor.
    :return: clip's loss on the batch.
    """
    batch = list(zip(*batch))
    imgs, texts = batch[0], batch[1]

    images = []
    for i in imgs:
        images.append(Image.open(i))

    logits_per_image,logits_per_text=calc_logits(texts, images,device,model,config,processor)

    return clip_loss(logits_per_text)

def calc_val_loss(test_set,batch_size,device,model,config,processor):
    """
    :param test_set: list of lists of texts and image paths.
    :param batch_size: the size of each batch
    :param device: the device that all of the tensors should be on.
    :param model: instance of CLIPModel.
    :param config: instance of CLIPConfig.
    :param processor: instance of CLIPProcessor.
    :return: clip's mean loss on test_set.
    """
    test_loss=0
    for _ in range(0, len(test_set), batch_size):
        batch = test_set[_:_ + batch_size]
        test_loss+=calc_loss(batch,device,model,config,processor).item()
    return test_loss/(len(test_set)/batch_size)
