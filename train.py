import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet,UNet_2Plus,UNet3Plus,U_Net,AttU_Net,R2U_Net,R2AttU_Net,DeepLab,Unetnew
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    #---------------------------------#
    # Cuda Whether to use Cuda
    # No GPU can be set to False
    #---------------------------------#
    Cuda = True
    #---------------------------------------------------------------------#
    # distributed is used to specify whether or not to use single multi-card distributed operation
    # Terminal commands are only supported in Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # DP mode is used to invoke all graphics cards by default under Windows. and DDP is not supported.
    # DP mode:
    # Set distributed = False.
    # Type CUDA_VISIBLE_DEVICES=0,1 in the terminal python train.py
    # DDP mode:
    # Set distributed = True
    # Type CUDA_VISIBLE_DEVICES=0,1 in terminal python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    # sync_bn whether to use sync_bn, DDP mode multicard available
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    # fp16 whether to use mixed precision training
    # fp16 whether to use mixed-precision training, reduces video memory by about half, requires pytorch 1.7.1+.
    # ---------------------------------------------------------------------#.0
    fp16 = False
    # -----------------------------------------------------#.0
    # num_classes must be modified to train your own dataset.
    # The number of categories you need +1, e.g. 2+1.
    #-----------------------------------------------------#
    num_classes = 2
    #-----------------------------------------------------#
    #   Backbone network selection
    #-----------------------------------------------------#
    backbone    = "resnest50"
    #----------------------------------------------------------------------------------------------------------------------------#
    # pretrained Whether to use the pretrained weights of the backbone network, here the weights of the backbone are used, so they are loaded during model construction.
    # If model_path is set, the weights of the backbone do not need to be loaded and the value of pretrained is meaningless.
    # If model_path is not set, pretrained = True, at which point only the trunk is loaded to start training.
    # If model_path is not set, pretrained = False, Freeze_Train = Fasle, at this point training starts from 0 and there is no process of freezing the trunk.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    # Pre-training weights for the model The more important part is the weights part of the backbone feature extraction network, which is used for feature extraction.
    # model_path
    # If there is an interruption in the training process, you can set the model_path to the weights file in the logs folder to reload the weights that have already been partially trained.
    # # Also modify the parameters of the freezing phase or unfreezing phase below to ensure the continuity of the model epoch.
    #
    # Do not load the entire model's weights when model_path = ''.
    #
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = ""
    #-----------------------------------------------------#
    #   input_shape     Enter the size of the image, in multiples of 32
    #-----------------------------------------------------#
    input_shape = [256, 256]
    #----------------------------------------------------------------------------------------------------------------------------#
    # Training is divided into two phases, the freeze phase and the unfreeze phase. The freeze phase is set to meet the training needs of students with underperforming machines.
    # Freeze training requires less video memory, and in the case of a very poor video card, Freeze_Epoch can be set equal to UnFreeze_Epoch, at which point just freeze training is performed.
    #
    # Provide here a number of parameter setting suggestions, trainers according to their own needs for flexible adjustment:
    #   (i) Start training with pre-training weights for the entire model:
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   (ii) Start training from the pre-training weights of the backbone network:
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    # Freeze phase training parameters
    # The backbone of the model is frozen at this point and the feature extraction network is not changed
    # Occupy less memory, only fine-tune the network.
    # Init_Epoch The current training generation of the model, its value can be greater than Freeze_Epoch, as set:
    # Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    # Will skip the freeze phase and start directly from generation 60, and adjust the corresponding learning rate.
    # (to be used when breaking the freeze)
    # Freeze_Epoch model freeze training for Freeze_Epoch
    # (disabled when Freeze_Train=False)
    # Freeze_batch_size batch_size for model freeze training
    # (Expires when Freeze_Train=False)

    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2
    #------------------------------------------------------------------#
    # Training parameters for the unfreezing phase
    # The backbone of the model is not frozen at this point, the feature extraction network changes
    # The memory used is larger and all the parameters of the network are changed
    # UnFreeze_Epoch Total number of epochs the model has been trained for.
    # Unfreeze_batch_size batch_size of the model after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    # Freeze_Train whether to do freeze training or not
    # Default to freeze the trunk training first and then unfreeze the training.
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    # Other training parameters: learning rate, optimizer, learning rate drop related to
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    # Maximum learning rate for Init_lr models
    # Init_lr=1e-4 recommended when using Adam optimizer
    # Init_lr=1e-2 recommended when using the SGD optimizer
    # Min_lr Minimum learning rate for the model, defaults to 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type the type of optimizer to use, optional adam, sgd
    # Init_lr=1e-4 is recommended when using the Adam optimizer.
    # Init_lr=1e-4 when using the SGD optimizer.
    # momentum The momentum parameter is used internally by the optimizer.
    # weight_decay weight_decay to prevent overfitting
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    # lr_decay_type Learning rate descent method used to, options are 'step', 'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    # ------------------------------------------------------------------#
    # save_period how many epochs to save weights once
    # ------------------------------------------------------------------#
    save_period = 5
    # ------------------------------------------------------------------#
    # save_dir Folder where weights and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # eval_flag Whether to evaluate at training time, the evaluation object is the validation set.
    # eval_period represents how many epochs to evaluate, frequent evaluation is not recommended.
    # eval_period means how many epochs to evaluate, frequent evaluation is not recommended.
    # The mAP obtained here will be different from the one obtained from get_map.py for two reasons:
    # (i) The mAP obtained here is the mAP of the validation set.
    # (a) The mAP obtained here is the mAP of the validation set. # (b) The evaluation parameters here are set more conservatively to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = False
    eval_period         = 5
    
    #------------------------------#
    #   数据集路径
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#

    dice_loss       = True
    #------------------------------------------------------------------#
    # Whether to use focal loss to prevent positive and negative sample imbalance
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    # Whether to assign different loss weights to different classes, the default is balanced.
    # If set, note that it is set to numpy form with the same length as num_classes.
    # As:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    # num_workers is used to set whether to use multi-threading to read data, 1 means turn off multi-threading.
    #------------------------------------------------------------------#
    num_workers     = 4

    #------------------------------------------------------#
    # Setting up the graphics card to be used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    #----------------------------------------------------#
    # Download pre-training weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)
    ######################################################################
    ######################################################################
    # model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    # model= UNet_2Plus(img_ch = 3,num_classes = num_classes).train()
    # model = UNet3Plus(n_channels=3, n_classes=num_classes, bilinear=True, feature_scale=4,is_deconv=True, is_batchnorm=True).train()
    # model = U_Net(img_ch=3, output_ch=num_classes).train()
    # model = AttU_Net(img_ch=3, output_ch=num_classes).train()
    # model = R2U_Net(img_ch=3, output_ch=num_classes, t=2).train()
    # model = DeepLab(num_classes=num_classes, backbone="xception", downsample_factor=8, pretrained=False).train()
    # model = R2AttU_Net(img_ch=3, output_ch=num_classes, t=2).train()
    model = Unetnew(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train() ##一组消融
    ######################################################################
    ######################################################################

    # if not pretrained:
    #     weights_init(model)
    # model.backbone.load_state_dict(torch.load('./model_data/pretrain.pth'), strict=False)
    if model_path != '':
        #------------------------------------------------------#
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        # Load based on Key of pre-training weights and Key of models
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        # Show Keys with no matches
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m Warmly, it's normal for the HEAD section not to load, and it's an error for the BACKBONE section not to load.\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    # torch 1.2 does not support amp, we recommend using torch 1.7.1 and above to use fp16 correctly.
    # So torch 1.2 shows "could not be resolved" here.
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    # Multi-card synchronization Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            # Multi-card parallel operation
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    # Read the txt corresponding to the dataset
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    # Backbone features extract network features generically, freezing training can speed up training
    # It can also prevent the weights from being corrupted at the beginning of training.
    # Init_Epoch is the starting generation.
    # Interval_Epoch is the frozen training epoch.
    # Epoch total training generations
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#

        if Freeze_Train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        # Set batch_size directly to Unfreeze_batch_size if you don't freeze training
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        # Determine current batch_size, adaptively adjust learning rate
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        # Select optimizer based on optimizer_type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        # Access to formulas for decreasing learning rates
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        # Judging the length of each generation
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, please expand the dataset.")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)
        
        #----------------------#
        # Record the map curve of eval
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        # Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            # If the model has a frozen learning component
            # Then unfreeze and set the parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                # Determine current batch_size, adaptively adjust learning rate
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                # Access to formulas for decreasing learning rates
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
