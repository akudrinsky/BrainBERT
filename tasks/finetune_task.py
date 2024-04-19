#usage
#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=False ++exp.runner.num_workers=0 +data=timestamp_data +model=debug_finetune_model ++exp.task.name=debug_finetune_task ++exp.criterion.name=debug_finetune_criterion ++exp.runner.total_steps=1000 ++model.frozen_upstream=True ++exp.runner.checkpoint_step=-1
import logging
import numpy as np
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import finetune_collator
from util.tensorboard_utils import plot_tensorboard_line
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)

@register_task(name="finetune_task")
class FinetuneTask(BaseTask):
    def __init__(self, cfg):
        super(FinetuneTask, self).__init__(cfg)

    def build_model(self, cfg):
        return models.build_model(cfg)
        
    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss": 0}
        predicts, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch["input"] = batch["input"].to(device)
                _, valid_outs = criterion(model, batch, device, return_predicts=True)

                # Assuming valid_outs["predicts"] contains raw logits or probabilities
                # Convert these to class predictions
                pred_classes = np.argmax(valid_outs["predicts"], axis=1)
                predicts.extend(pred_classes)
                
                labels.extend(batch["labels"])
                all_outs["loss"] += valid_outs["loss"]

        # Convert lists to NumPy arrays for sklearn compatibility
        predicts = np.array(predicts)
        labels = np.array(labels)

        # Calculate accuracy
        accuracy = accuracy_score(labels, predicts)
        all_outs["loss"] /= len(valid_loader)
        all_outs["accuracy"] = accuracy  # Store accuracy in all_outs dictionary
        return all_outs

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, collate_fn=finetune_collator, **kwargs)

    def output_logs(self, train_logging_outs, val_logging_outs, writer, global_step):
        val_auc_roc = val_logging_outs["accuracy"]
        if writer is not None:
            writer.add_scalar("accuracy", val_auc_roc, global_step)
        log.info(f'accuracy: {val_auc_roc}')

        # image = train_logging_outs["images"]["wav"]
        # label = train_logging_outs["images"]["wav_label"]
        # tb_image = plot_tensorboard_line(image, title=label)
        # if writer is not None:
            # writer.add_image("raw_wave", tb_image, global_step)
