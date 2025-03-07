import lightning.pytorch as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
import logging
import os


logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.getLogger('nemo_logging').setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)


@hydra_runner(
    config_path="/content/jarvis_stt_dataset3/jarvis_stt_dataset/scripts/configs", config_name="fastconformer_hybrid_transducer_ctc_bpe_colab"
)
def main(cfg):
    print(1)
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)


if __name__ == '__main__':
  main()
