# coding = utf-8
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from Continuous_VAE.model.cvae import ContinuousAgent, ContinuousVAE
from Continuous_VAE.data_apis.data_utils import DataUtils
from Continuous_VAE.run_config import RunConfig as Config

if __name__ == "__main__":
    config = Config()
    api = DataUtils(config.ctx_encode_method)
    api.load_vocab()
    api.load_candidates()
    api.load_dialog(config.coming_task, config.system_mode)
    api.build_pad_config(config.memory_size)

    model = ContinuousVAE(config, api)

    if config.trained_model is not None:
        if os.path.isdir(config.trained_model):
            trained_model = os.listdir(config.trained_model)
            trained_model.sort(key=lambda x: int(x.split("_")[1]))
            config.trained_model = os.path.join(config.trained_model, trained_model[-1])
        print("Using trained model in {}".format(config.trained_model))
        model.load_state_dict(torch.load(config.trained_model))

    agent = ContinuousAgent(config, model, api)
    agent.main()
