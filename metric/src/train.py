from pathlib import Path 
from argparse import ArgumentParser
import shutil 

from trainers import get_trainer
from dataset import get_dataset
from models import get_model 

from utils import setup_logger, read_yaml

def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        '-o', '--result_dir',
        help='Result directory name (not path)',
        required=True,
        type=str
    )
    parser.add_argument(
        '-i', '--param',
        required=True,
        type=str
    )

    parser.add_argument(
        '-d', '--dataset',
        required=True,
        type=str
        
    )
    parser.add_argument(
        '-g', '--device', 
        default='', 
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    ) 


    parser.add_argument(
        '--log_level',
        type=int,
        default=30,
        help='10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL'
    )
    return parser


def main(config, dset_config):

    dset = get_dataset(
        config, dset_config, mode="train"
    )
    valid_dset = get_dataset (
        config, dset_config, mode="valid"
    )
    model = get_model(
        config, dset_config
    )

    trainer = get_trainer(
        config, dset_config
    )
    trainer.train(
        dataset=dset,
        valid_dataset=valid_dset,
        model=model
    )

    trainer.save()

if __name__=="__main__":
    parser = argparser()
    args   = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    config = read_yaml(args.param)
    dset_config = read_yaml(args.dataset)
    breakpoint()
    setattr(config, "device", args.device) 
    setattr(config, "result_dir", args.result_dir)
    main(config, dset_config)

else:
    logger = setup_logger(__name__)