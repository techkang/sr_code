import os

import trainer
from config import setup, default_argument_parser, init_logging
from tools.launch import launch_job


def main(args, config):
    init_logging(args, config)

    if not config.eval_only:
        getattr(trainer, config.trainer)(config, resume=args.resume).train()
    else:
        trainer_test = getattr(trainer, config.trainer)(config, resume=args.resume)
        trainer_test.test()


if __name__ == '__main__':
    arg = default_argument_parser().parse_args()
    if arg.visible_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.visible_gpus
    cfg = setup(arg)
    launch_job(main, arg, cfg)
