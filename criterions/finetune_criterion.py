import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion

@register_criterion("finetune_criterion")
class FinetuneCriterion(BaseCriterion):
    def __init__(self):
        super(FinetuneCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, model, batch, device, return_predicts=False):
        inputs = batch["input"].to(device) #potentially don't move to device if dataparallel
        pad_mask = batch["pad_mask"].to(device)

        output = model.forward(inputs, pad_mask)
        labels = torch.LongTensor(batch["labels"]).to(output.device)
        loss = self.loss_fn(output, labels)
        images = {"wav": batch["input"][0],
                  "wav_label": batch["labels"][0]}
        if return_predicts:
            predicts = self.softmax(output).detach().cpu().numpy()
            logging_output = {"loss": loss.item(),
                              "predicts": predicts,
                              "images": images}
        else:
            logging_output = {"loss": loss.item(),
                              "images": images}
        return loss, logging_output
