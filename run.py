from DataDamloader import *
from DataDamSolver import *
from ResNetAndrewNg import *
import pandas as pd
from torch.utils.data import DataLoader


def main():
    config = get_config()
    solver = DataDamSolver(config)
    solver.fit()


def get_config():
    # 학습할 configuration을 세팅합니다.
    config = dict()

    # 실행시킬 라벨을 입력하세요
    config["data"] = dict()
    config["data"]["train"] = "../label/DATADAM_sleep_train.csv"
    config["data"]["valid"] = "../label/DATADAM_sleep_valid.csv"
    config["data"]["test"] = "../label/DATADAM_sleep_test.csv"

    # 모델을 세팅하세요
    config["kernel_size"] = 5
    config["dropout"] = 0.15
    config["output_size"] = 2
    config["model"] = ResNetAndrewNg(config).cuda()

    config["dataloader"] = dict()
    for k, v in config["data"].items():
        table = pd.read_csv(v)
        shuffle = True if k == "train" else False
        config["dataloader"][k] = DataLoader(
            dataset=DataDamloader(table),
            batch_size=4,
            shuffle=shuffle,
            drop_last=False,
        )

    # loss function을 세팅하세요
    config["loss_function"] = torch.nn.CrossEntropyLoss()

    # optimizer를 세팅하세요
    config["optimizer"] = torch.optim.Adam(
        config["model"].parameters(), lr=0.0001, weight_decay=0.00001
    )

    # epoch을 세팅하세요
    config["epochs"] = 2

    # save_path를 세팅하세요
    config["save_path"] = "../result"

    return config


if __name__ == "__main__":
    main()
