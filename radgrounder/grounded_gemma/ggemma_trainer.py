from trl import SFTTrainer
import torch

class GroundedGemmaSFTTrainer(SFTTrainer):
    """Custom SFT Trainer that logs segmentation and LM losses separately"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if "lm_loss" in outputs:
            self._metrics[mode]["lm_loss"].append(outputs.lm_loss.item()) if hasattr(outputs, 'lm_loss') else None
        else:
            self._metrics[mode]["lm_loss"] = [outputs.lm_loss.item()]

        if outputs.seg_loss is not None:
            if "seg_loss" in outputs:
                self._metrics[mode]["seg_loss"].append(outputs.seg_loss.item()) if hasattr(outputs, 'seg_loss') else None
            else:
                self._metrics[mode]["seg_loss"] = [outputs.seg_loss.item()]

        new_losses = {
            "seg_loss": self._metrics[mode]["lm_loss"],
            "lm_loss": self._metrics[mode]["seg_loss"]
        }
        # self.log(new_losses)
        # seg_loss = outputs.seg_loss if hasattr(outputs, 'seg_loss') else None
        # self._metrics[mode]["segmentation_loss"].append(seg_loss)


        return (loss, outputs) if return_outputs else loss