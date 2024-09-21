from transformers import CLIPModel
import torch
from typing import Optional, Tuple


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(logits_per_text: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(logits_per_text)
    image_loss = contrastive_loss(logits_per_text.T)
    return (caption_loss + image_loss) / 2.0


class ClipMDModel(CLIPModel):

    def embed_text(self,
                   input_ids:torch.LongTensor,
                   attention_mask:torch.LongTensor,
                   output_attentions: Optional[bool] = None,
                   output_hidden_states: Optional[bool] = None,
                   position_ids: Optional[torch.LongTensor] = None,
                  ):
        """
        :param input_ids: tokenized text from CLIPProcessor.
        :param attention_mask: attention mask from CLIPProcessor.
        :return: text embeddings of input_ids (tokens longer then 77 tokens
        is embeded using a sliding window and pooling).
        """
        tokens = []
        masks = []
        pos = []

        for i in range(input_ids.size()[0]):
            ten = input_ids[i]
            mask = attention_mask[i]
            mask = mask[mask.nonzero().flatten()]
            ten = ten[:mask.size()[0]]

            if not pos:
                pos.append([0, 0])
            else:
                pos.append([pos[-1][1], pos[-1][1]])

            #spliting tokenized text into input sized chunks with an overlapping window.
            if ten.size()[0]>77:
                tokens.append(ten.unfold(dimension = 0,size = 77, step = 70))
                masks.append(mask.unfold(dimension = 0,size = 77, step = 70))

                pos[-1][1]+=tokens[-1].size()[0]

                ten=ten[tokens[-1].size()[0]*70:]
                mask=mask[tokens[-1].size()[0]*70:]

            if ten.size()[0] > 0:
                new_mask = torch.zeros((1, 77)).to(self.device)
                new_mask[:, 0:mask.size()[0]] = mask

                new_ten = torch.full((1, 77), 49407).to(self.device)
                new_ten[:, 0:ten.size()[0]] = ten

                tokens.append(new_ten)
                masks.append(new_mask)
                pos[-1][1] += 1
        #encoding the tokenized text
        embedded = self.get_text_features(input_ids=torch.cat(tokens, 0),
                                          attention_mask=torch.cat(masks, 0),
                                         output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states,
                                         position_ids=position_ids,
                                         )
        
        #pooling the embeddings of segments that came from the same original text
        embeddings = []
        for p in pos:
            if p[1] - p[0] == 1:
                embeddings.append(embedded[p[0]].unsqueeze(0))
            else:
                embeddings.append(torch.mean(embedded[p[0]:p[1]], dim=0).unsqueeze(0))

        return torch.cat(embeddings, 0)

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        """
        :param input_ids: tokenized text from CLIPProcessor.
        :param attention_mask: attention mask from CLIPProcessor.
        :param pixel_values: pixel values from CLIPProcessor.
        :param return_loss: boolean that indicates if loss should be returned
        :return: image-caption cosine similarity as logits per image and per caption (also loss if return_loss is true)
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = self.config.use_return_dict
        
        #encoding the images 
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        #encoding the text captions
        text_embeds =self.embed_text(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     position_ids=position_ids
        )


        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        if return_loss:
            loss = clip_loss(logits_per_text)
            return logits_per_image,logits_per_text,loss
        return logits_per_image,logits_per_text
