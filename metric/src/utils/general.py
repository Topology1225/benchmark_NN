import os
import glob
from pathlib import Path
from typing import Union, List
import json
import re
import shutil

import yaml
from attrdict import AttrDict

from utils.logging import setup_logger


logger = setup_logger(__name__)


def read_yaml(path: List[Union[str, list]]) -> AttrDict:
    """yamlを読み込み, dictのkeyをattrとするインスタンスをreturn
    Parameters
    ----------
    path: str or list


    Return: obj
        AttrDict
    """
    if isinstance(path, str):
        obj = _read_yaml(path)
    elif isinstance(path, list):
        obj = dict()
        for p in path:
            assert isinstance(p, str)
            _obj = _read_yaml(p)
            if __debug__:
                for key in _obj.keys():
                    assert not key in obj.keys(), f"{key} は他のconfigで設定されてます."

            obj.update(_obj)
    obj = AttrDict(obj)
    return obj


def _read_yaml(path: str) -> dict:
    """.yaml fileを読み込む関数
    Return:
        obj: dict
    """
    logger.debug(f"\n [ READ ] {path}")
    f = open(path, mode="r")
    obj = yaml.safe_load(f)
    f.close()

    for key, value in obj.items():
        logger.info(f"\n {key} ← {value}")
    return obj


def increment_path(path, exist_ok=False, sep=""):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def save2json(x: dict, save_path: str) -> None:
    f = open(save_path, mode="w")
    json.dump(x, fp=f)
    f.close()


def save_yaml(config):
    param_name = config.param_path.split("/")[-1]
    save_param_pt = os.path.join(config.result_dir, param_name)
    logger.debug(f"\n[SAVE]: {config.param_path}→{save_param_pt}")
    shutil.copy(config.param_path, save_param_pt)

    param_name = config.dset_param_path.split("/")[-1]
    save_param_pt = os.path.join(config.result_dir, param_name)
    logger.debug(f"\n[SAVE]: {config.dset_param_path}→{save_param_pt}")
    shutil.copy(config.dset_param_path, save_param_pt)


def save_hostname(config):
    import socket

    hostname = socket.gethostname()
    file_name = os.path.join(config.result_dir, f"{hostname}.txt")
    logger.info(f"\n {hostname}")
    with open(file_name, mode="w") as f:
        pass
