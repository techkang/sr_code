import logging

import loss


def build_loss(cfg):
    loss_name = cfg.solver.loss
    if ',' in loss_name:
        logging.info('using multi loss')
        loss_name = loss_name.split(',')
        losses = [getattr(loss, i) for i in loss_name]
        weights = cfg.solver.loss_weights

        def loss_func(encoder, decoder, reduction='mean'):
            loss_dict = {}
            assert len(losses) == len(weights), f'not enough weights given with loss:{loss_name}, weights:{weights}'
            for every_loss, every_name, every_weight in zip(losses, loss_name, weights):
                loss_dict.update({every_name: every_loss(encoder, decoder, reduction) * every_weight})
            return loss_dict

        return loss_func

    else:
        return getattr(loss, loss_name)
