from typing import Union, List

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
                    assert not key in obj.keys(),\
                        f"{key} は他のconfigで設定されてます."
            
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