import logging
import os


def log(msg):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(msg)


def log_parameter(opt, device):
    msg = '\033[1;34mparameter\033[0m: epochs=' + str(opt.epochs) + ',batch_size=' + str(opt.batch_size) + \
          ',lr=' + str(opt.lr) + ',device=' + str(device) + ',chinese=' + opt.chinese + ',images=' + opt.images + \
          ',labels' + opt.labels + ',imgH=' + str(opt.imgH) + ',nc=' + str(opt.nc) + ',nh=' + str(opt.nh) + \
          ',val_epoch=' + str(opt.val_epoch) + ',save_all=' + str(opt.save_all) + ',best=' + str(opt.best) + \
          ',test=' + str(opt.test) + ',all=' + str(opt.all) + ',weights=' + opt.weights + ',name=' + opt.name
    log(msg)


def log_model(model):
    msg = '\033[1;34mmodel\033[0m:\n' + str(model)
    log(msg)


def log_optimizer(optimizer):
    msg = '\033[1;34moptimizer\033[0m:\n' + str(optimizer)
    log(msg)


def log_epoch(epoch, loss, acc, mode):
    msg = 'epoch {}: {}_loss={}, {}_acc={}'.format(epoch, mode, loss, mode, acc)
    log(msg)


def log_save_model(epoch, loss, acc, flag='chinese'):
    msg = '\033[1;32msave model {}_{}_{}_{}.pt finish, the saved path is weights/\033[0m'.format(flag, epoch, loss, acc)
    log(msg)


def log_load_model(weight, mode='train'):
    msg = ''
    if os.path.exists(weight):
        msg = 'load {} model success!'.format(weight)
    elif mode == 'train':
        msg = 'no saved model, training will start from scratch!'
    elif mode == 'test' or mode == 'detect':
        msg = 'load model fail!'
    log(msg)


def log_test(loss, acc, batch=True):
    if batch:
        msg = 'test_batch_loss={}, test_batch_acc={}'.format(loss, acc)
    else:
        msg = 'test_loss={}, test_acc={}'.format(loss, acc)
    log(msg)


def log_device(device):
    msg = 'device: {}'.format(str(device))
    log(msg)
