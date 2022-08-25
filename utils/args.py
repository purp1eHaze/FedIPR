import argparse

def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= federated learning parameters ========================
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr_outer', type=float, default=1,
                        help="learning rate")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate for inner update")
    parser.add_argument('--gamma', type=float, default=0.99,
                         help="exponential weight decay")
    parser.add_argument('--iid', action='store_true', default =False,
                        help='dataset is split iid or not')
    parser.add_argument('--wd', type=float, default=4e-5,
                        help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='communication round')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of client sampling we use') 
    
    # ============================ Model arguments ===================================
    parser.add_argument('--model_name', type=str, default='alexnet', choices=['alexnet', 'resnet'],
                        help='model architecture name')
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help="name of dataset")
    
    parser.add_argument('--data_root', default='/home/lbw/Data/',
                        help='dataset directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--bp_interval', default=30, type=int, help='interval for starting bp the local part')
    parser.add_argument('--log_interval', default=1, type=int,
                        help='interval for evaluating loss and accuracy')

  
    # =========================== IPR parameters ===================================
    
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'gn', 'in', 'none'],
                        help='norm type (default: bn)')
    parser.add_argument('--key-type', choices=['random', 'image', 'shuffle'], default='shuffle',
                        help='passport key type (default: shuffle)')
    # signature argument

    parser.add_argument('--num_sign', type=int, default=1,
                        help="number of signature users: K")

    parser.add_argument('--weight_type', default='gamma', choices=['gamma', 'kernel'],
                        help='weight-type (default: gamma)')
    
    parser.add_argument('--num_bit', type=int, default=20,
                        help="number of signature bits")

    parser.add_argument('--loss_type', default='sign', choices=['sign', 'CE'],
                        help='loss type (default: sign)')

    parser.add_argument('--loss_alpha', type=float, default= 0.2,
                        help='sign loss scale factor to trainable (default: 0.2)')

    # backdoor argument 
    parser.add_argument('--backdoor_indis', action='store_false', default=True,
                        help='backdoor in distribution')
    parser.add_argument('--num_back', type=int, default=1,
                        help="number of backdoor users: K")
    parser.add_argument('--num_trigger', type=int, default=40,
                        help="number of signature bits")

    # paths
    parser.add_argument('--passport-config', default='passport_configs/alexnet_passport.json',
                        help='should be same json file as arch')

    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')

    # =========================== DP ===================================
    parser.add_argument('--dp', action='store_true', default=False,
                        help='whether dp')

    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')



    # =========================== Robustness ===================================
    parser.add_argument('--pruning', action='store_true')
    parser.add_argument('--percent', default=5, type=float)

    # parser.add_argument('--im_balance', action='store_true', default=False,
    #                     help='whether im_balance')
    
    args = parser.parse_args()

    return args
