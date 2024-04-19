from models import register_model
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.masked_tf_model import MaskedTFModel

@register_model("classification_model")
class ClassificationModel(BaseModel):
    def __init__(self):
        super(ClassificationModel, self).__init__()

    def _val_nans(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in {name}")
                return True
        return False

    def build_model(self, cfg):
        self.backbone = MaskedTFModel()
        self.backbone.build_model(cfg)
        self.backbone.load_state_dict(torch.load(cfg.upstream_ckpt)['model'])
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.cfg = cfg
        assert not self._val_nans()

    def forward(self, input_specs, src_key_mask):
        initial_shape = input_specs.shape

        assert not torch.isnan(input_specs).any(), torch.isnan(input_specs).sum()
        assert not torch.isnan(src_key_mask).any(), torch.isnan(src_key_mask).sum()
        output_specs = self.backbone(input_specs, src_key_mask, True, -1)
        output_specs = self.fc(output_specs)
        # print(output_specs.shape) # (B, T, N_CLASSES)

        output_logits = output_specs.mean(dim=1)

        return output_logits


