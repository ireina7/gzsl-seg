import torch
import torch.nn as nn
import numpy as np # type: ignore
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import matplotlib.pyplot as plt # type: ignore

from src.config import *
from util import *
from util.typing.basic import *
from util.syntax import *
from dataset.voc.dataset_voc import dataloader_voc
from model.vgg_voc import Our_Model
from util.typing.torch import DataLoader, Tensor
# from util.typing.typeclass import instances




def load_model(args) -> Tuple[Our_Model, int]:
  if args.restore_from_where == "pretrained":
    return Our_Model(split), 0
  else:
    logger.error('Restore model function has not been supported!')
    # restore_from = get_model_path(args.snapshot_dir)
    # model_restore_from = restore_from["model"]
    # i_iter = restore_from["step"]

    # model = Our_Model(split)
    # saved_state_dict = torch.load(model_restore_from)
    # model.load_state_dict(saved_state_dict)
end(load_model)


def eval_model(args, model, mode: Mode) -> float:
    device = args.device
    w, h = args.input_size.split(",")
    input_size = (int(w), int(h))
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear", align_corners=True)
    eval_loader = dataloader_voc(split = split, mode = mode)
    data_len = len(eval_loader)

    iter = enumerate(eval_loader)
    model.eval()
    hist = np.zeros((15, 15))
    average_mIoUs_per_epoch = 0
    logger.debug(data_len, 'eval data length')
    for i in range(data_len):
        logger.blank_line()
        logger.log('> Evaluation step {}'.format(i))
        _, batch = iter.__next__()
        images, masks = batch["image"], batch["label"]
        images = images.to(device)
        masks = masks.long().to(device)
        pred = model(images, "all")
        pred = interp(pred)

        max_ = torch.argmax(pred, 1)
        pred_IoU = max_[0].clone().detach().cpu().numpy()

        # debug(pred_IoU.shape, 'pred_IoU.shape')
        #pred_cpu = pred_IoU.data.cpu().numpy()
        pred_cpu = pred_IoU
        mask_cpu = masks[0].cpu().numpy()

        pred_cpu[mask_cpu == 255] = 255
        m = confusion_matrix(mask_cpu.flatten(), pred_cpu.flatten(), 15)
        #hist += m
        hist += m
        mIoUs = per_class_iu(hist)
        average_mIoUs = sum(mIoUs) / len(mIoUs)
        average_mIoUs_per_epoch = average_mIoUs
        
        logger.log("> mIoU: \n{}".format(per_class_iu(m)))
        logger.log("> mIoUs: \n{}".format(mIoUs))
        logger.log("> Average mIoUs: \n{}".format(average_mIoUs))

        if i % 100 == 0:
            ans = max_[0].clone().detach().cpu().numpy()
            #x = np.where(ans == 0, 255, ans)
            x = ans
            y = masks.cpu()[0]
            x[y == 255] = 255
            painter.draw_sample(batch)
            # pyplot.figure()
            plt.imshow(x, cmap = 'tab20', vmin = 0, vmax = 21)
            plt.colorbar()
            # show_figure_nonblocking()
            logger.save_figure('output/Eval-{}-{}.pdf'.format(i, batch['name'][0]))
        #end for
    #end eval_model


def eval_model_unseen(args, model, mode: Mode) -> float:
    device = args.device
    w, h = args.input_size.split(",")
    input_size = (int(w), int(h))
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear", align_corners=True)
    eval_loader = dataloader_voc(split = split, mode = mode)
    data_len = len(eval_loader)

    iter = enumerate(eval_loader)
    model.eval()
    hist = np.zeros((15, 15))
    average_mIoUs_per_epoch = 0
    logger.debug(data_len, 'eval data length')
    for i in range(data_len):
        logger.blank_line()
        logger.log('> Evaluation step {}'.format(i))
        _, batch = iter.__next__()
        images, masks = batch["image"], batch["label"]
        images = images.to(device)
        masks = masks.long().to(device)
        pred = model(images, "all")
        pred = interp(pred)

        max_ = torch.argmax(pred, 1)
        pred_IoU = max_[0].clone().detach().cpu().numpy()

        # debug(pred_IoU.shape, 'pred_IoU.shape')
        #pred_cpu = pred_IoU.data.cpu().numpy()
        pred_cpu = pred_IoU
        mask_cpu = masks[0].cpu().numpy()

        pred_cpu[mask_cpu == 255] = 255
        m = confusion_matrix(mask_cpu.flatten(), pred_cpu.flatten(), 15)
        #hist += m
        hist += m
        mIoUs = per_class_iu(hist)
        average_mIoUs = sum(mIoUs) / len(mIoUs)
        average_mIoUs_per_epoch = average_mIoUs
        
        logger.log("> mIoU: \n{}".format(per_class_iu(m)))
        logger.log("> mIoUs: \n{}".format(mIoUs))
        logger.log("> Average mIoUs: \n{}".format(average_mIoUs))
        

        if i % 100 == 0:
            ans = max_[0].clone().detach().cpu().numpy()
            #x = np.where(ans == 0, 255, ans)
            x = ans
            y = masks.cpu()[0]
            x[y == 255] = 255
            painter.draw_sample(batch)
            # pyplot.figure()
            plt.imshow(x, cmap = 'tab20', vmin = 0, vmax = 21)
            plt.colorbar()
            # show_figure_nonblocking()
            logger.save_figure('output/Eval-{}-{}.pdf'.format(i, batch['name'][0]))
        #end for
    #end eval_model








def main(args=get_arguments()) -> None:
  """ 
  Main zero shot segmentation function 
  @author Ireina7
  @date 2021.07
  """
  device = args.device
  print_config(args)

  input_size = args.input_size
  # input_size = (int(w), int(h))
  model, i_iter = load_model(args)
  model.train()
  model.to(device)

  if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)

  train_loader = dataloader_voc(split = split)
  data_len = len(train_loader)
  num_steps = data_len * args.num_epochs

  
  optimizer = optim.SGD(
    model.optim_parameters_1x(args),
    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  
  optimizer_10x = optim.SGD(
    model.optim_parameters_10x(args),
    lr=10 * args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
  # optimizer = optim.Adam(
  #     model.optim_parameters_1x(args),
  #     lr=args.learning_rate, weight_decay=args.weight_decay)
  #
  # optimizer_10x = optim.Adam(
  #     model.optim_parameters_10x(args),
  #     lr=10 * args.learning_rate, weight_decay=args.weight_decay)

  optimizer.zero_grad()
  optimizer_10x.zero_grad()

  seg_loss = nn.CrossEntropyLoss(ignore_index=255)
  #seg_loss = FocalLoss() # merely test if focal loss is useful...

  interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear", align_corners=True)

  with open(RESULT_DIR, "a") as f:
    f.write(SNAPSHOT_PATH.split("/")[-1] + "\n")
    f.write("lambda : " + str(lambdaa) + "\n")

  average_mIoUs_history = []

  logger.blank_line(2)
  logger.log('Training start ...')
  for epoch in range(args.num_epochs + 1):
    # blank_line()
    logger.log(">> Epoch: {}".format(epoch))
    train_iter = enumerate(train_loader)
    model.train()
    hist = np.zeros((15, 15))
    average_mIoUs_per_epoch = 0
    for i in range(data_len):
      logger.blank_line()
      logger.log("> Epoch {}, loop {}".format(epoch, i))
      loss_pixel = 0
      loss_pixel_value = 0

      optimizer.zero_grad()
      adjust_learning_rate(optimizer, i_iter, num_steps, args, times=1)

      optimizer_10x.zero_grad()
      adjust_learning_rate(optimizer_10x, i_iter, num_steps, args, times=10)

      # train strong
      try:
        _, batch = train_iter.__next__()
      except StopIteration:
        train_strong_iter = enumerate(train_loader)
        _, batch = train_iter.__next__()

      images, masks = batch["image"], batch["label"]
      #print("mask: ", masks[0].min())
      images = images.to(device)
      masks = masks.long().to(device)
      pred = model(images, "all")
      pred = interp(pred)


      # Calculate mIoU
      logger.debug(pred.shape, 'pred.shape')
      max_ = torch.argmax(pred, 1)
      pred_IoU = max_[0].clone().detach().cpu().numpy()

      logger.debug(pred_IoU.shape, 'pred_IoU.shape')
      #pred_cpu = pred_IoU.data.cpu().numpy()
      pred_cpu = pred_IoU
      mask_cpu = masks[0].cpu().numpy()

      pred_cpu[mask_cpu == 255] = 255
      m = confusion_matrix(mask_cpu.flatten(), pred_cpu.flatten(), 15)
      #hist += m
      hist += m
      mIoUs = per_class_iu(hist)
      average_mIoUs = sum(mIoUs) / len(mIoUs)
      average_mIoUs_per_epoch = average_mIoUs
      
      if (EPOCH_TO_SHOW_MSG(epoch) and LOOP_TO_SHOW_MSG(i)):
        logger.log("> mIoU: \n{}".format(per_class_iu(m)))
        logger.log("> mIoUs: \n{}".format(mIoUs))
        logger.log("> Average mIoUs: \n{}".format(average_mIoUs))
      end
      
      loss_pixel = seg_loss(pred, masks)
      loss = loss_pixel# + loss_qfsl

      max_ = torch.argmax(pred, 1)
      #print(max_[0])

      if (EPOCH_TO_SHOW_FIGURE(epoch) and LOOP_TO_SHOW_FIGURE(i)):
        painter.draw_seg_result(batch, max_, masks),
        painter.save_figure(
          'output/Epoch-{}-{}-{}.pdf'
            .format(epoch, i, batch['name'][0])
        )
      end
      
      logger.log(f"loss: {loss}")
      loss.backward()
      optimizer.step()
      optimizer_10x.step()

    average_mIoUs_history.append(average_mIoUs_per_epoch)
    '''
    Save mIoU history
    '''
    if epoch > 0 and epoch % SHOW_EPOCH == 0:
      painter.plot(range(0, len(average_mIoUs_history)), average_mIoUs_history)
      # show_figure_nonblocking()
      if epoch >= SHOW_EPOCH:
        logger.save_figure(f'output/mIoUs of Epoches {0}-{epoch}.pdf')
      end
    end

    '''
    Evaluate model performance on val dataset
    '''
    if epoch >= 5 or epoch % 10 == 0:
      logger.blank_line(2)
      logger.log('>> Evaluation stage starting ...')

      logger.log('Evaluating in seen mode ...')
      eval_model(args, model, Mode.val_seen)

      logger.log('Evaluating in unseen mode ...')
      eval_model_unseen(args, model, Mode.val_unseen)
    #end if
  #end for epoch
#end main


# from util.typing.typeclass import *
# from util.experiment import *

# instance(DeepLearning[Our_Model], summon(Default[DeepLearning]).default())





if __name__ == "__main__":
    main()
