# coding = utf-8
import torch
import os
import sys

sys.path.append("/home/wkwang/workstation/experiment/src")
from Dual_rnn.model.dlstm import DualLSTM, DualLSTMAgent
from config.config import DATA_ROOT
from Dual_rnn.data_apis.data_utils import DataUtils, build_p_mapping
from Dual_rnn.run_config import RunConfig as Config


if __name__ == "__main__":
    config = Config()
    api = DataUtils()
    if config.system_mode == "train":
        train_file = os.path.join(DATA_ROOT, "public", config.trained_task, "train.txt")
        dev_file = os.path.join(DATA_ROOT, "public", config.trained_task, "dev.txt")
        candidates_file = os.path.join(DATA_ROOT, "candidate", config.trained_task + ".txt")
        api.load_vocab(config.trained_task)
        api.load_candidates(config.trained_task)
        train_data = api.load_dialog(train_file)
        dev_data = api.load_dialog(dev_file)
        data = train_data + dev_data
        api.build_pad_config(data)
        candidates = api.vectorize_candidates()
        model = DualLSTM(config, candidates).to(config.device)
        if config.trained_model is not None:
            if os.path.isdir(config.trained_model):
                trained_model = os.listdir(config.trained_model)
                trained_model.sort(key=lambda x: int(x.split("_")[1]))
                config.trained_model = os.path.join(config.trained_model, trained_model[-1])
            print("Using trained model in {}".format(config.trained_model))
            model.load_checkpoints(config.trained_model)
        agent = DualLSTMAgent(config,model,train_data, dev_data, None, api)
        agent.train()
    else:
        test_file = os.path.join(DATA_ROOT, "public", config.testing_task, "train.txt")
        if config.aware_new:
            api.load_vocab(config.testing_task)
            api.load_candidates(config.testing_task)

            original_api = DataUtils()
            original_api.load_vocab(config.trained_task)
            mapping_dict = {"embedding": build_p_mapping(original_api.word2index, api.word2index)}
        else:
            api.load_vocab(config.trained_task)
            api.load_candidates(config.trained_task)
            mapping_dict = {}
        test_data = api.load_dialog(test_file)
        api.build_pad_config(test_data)
        candidates = api.vectorize_candidates()

        model = DualLSTM(config,candidates)
        assert config.trained_model is not None
        if os.path.isdir(config.trained_model):
            trained_model = os.listdir(config.trained_model)
            trained_model.sort(key=lambda x: int(x.split("_")[1]))
            config.trained_model = os.path.join(config.trained_model, trained_model[-1])
        print("Using trained model in {}".format(config.trained_model))
        model.load_checkpoints(config.trained_model, mapping_dict)
        agent = DualLSTMAgent(config, model, None, None, test_data, api)
        agent.test()
