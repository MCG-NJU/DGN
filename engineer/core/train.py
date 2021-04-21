'''
@author:lingteng qiu
@name:OPEC_GCN
'''
import os
from utils.bar import Bar
from utils.structure import AverageMeter
import time
import torch.nn as nn
import torch
from engineer.core.eval import eval_map
from engineer.utils import lr_step_method as optim


def train_epochs(model_pos,optimizer,cfg,train_loader,pose_generator,criterion1, criterion2,test_loader,pred_json,writer_dict):
    best_map =None

    begin_epoch = 0

    # add resume function
    checkpoint_file = os.path.join(cfg.checkpoints, 'checkpoint.pth')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']+1
        best_map = checkpoint['map']
        writer_dict['train_step'] = checkpoint['train_step']
        writer_dict['valid_step'] = checkpoint['valid_step']
        model_pos.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    
    writer_map = open(os.path.join(cfg.checkpoints,"mAP.txt"),'w')
    for epoch in range(begin_epoch, cfg.nEpochs):
        if epoch > 0:
            print("=> start epoch {}, count from epoch 0".format(epoch))
        #average
        epoch_loss_2d_pos = AverageMeter()
        # epoch_loss_heat_map = AverageMeter()
        epoch_loss_score = AverageMeter()
        epoch_loss_edge = AverageMeter()
        epoch_loss = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        bar = Bar('Train', max=int(len(train_loader) // cfg.PRINT_FREQ)+1)
        for _,batches in enumerate(train_loader):
            inps, orig_img_list, img_name_list, boxes, scores, pt1, pt2, gts_list, dts_list = batches
            if pose_generator is not None:
                dts,gt_2d,hm_4,ret_features,heatmaps = pose_generator(inps,orig_img_list,img_name_list,boxes,scores,pt1,pt2,gts_list,dts_list)
                dts = dts.cuda()
                gt_2d = gt_2d.cuda()
                # hm_4 = hm_4.cuda()
                ret_features = [ret.cuda() for ret in ret_features]
                bz = dts.shape[0]
                data_time.update(time.time() - end)
                out_2d, edge_2d, edge_gt = model_pos(dts, heatmaps, ret_features, gt=gt_2d)
                # heat_map_regress = heat_map_regress.view(-1, 12, 2)
            else:
                out_2d, heat_map_regress, inter_gral_x, gt_2d, bz = model_pos(inps,orig_img_list,img_name_list,boxes,scores,pt1,pt2,gts_list,dts_list)
                heat_map_regress = heat_map_regress.view(-1, 12, 2)
                data_time.update(time.time() - end)
            lr = optim.get_epoch_lr(epoch + float(_) / len(train_loader), cfg)
            optim.set_lr(optimizer, lr)




            optimizer.zero_grad()
            labels = gt_2d[:,...,2]
            labels = labels[:,:,None].repeat(1,1,2)

            gt_edges = edge_gt[0]
            labels_edges = edge_gt[1]
            labels_edges = labels_edges.repeat(1,1,2)
            gt_edges = gt_edges[labels_edges > 0].view(-1, 2)

            gt_2d = gt_2d[:, ..., :2]
            gt_2d = gt_2d[labels > 0].view(-1, 2)
            out_2d_0 = out_2d[0][labels > 0].view(-1, 2)
            out_2d_1 = out_2d[1][labels > 0].view(-1, 2)
            out_2d_2 = out_2d[2][:,...,:2][labels > 0].view(-1, 2)
            out_score = out_2d[2][:,...,2][labels[:,...,0]>0]

            edge_2d_0 = edge_2d[0][labels_edges > 0].view(-1, 2)
            edge_2d_1 = edge_2d[1][labels_edges > 0].view(-1, 2)
            edge_2d_2 = edge_2d[2][:,...,:2][labels_edges > 0].view(-1, 2)
            edge_score = edge_2d[2][:,...,2][labels_edges[:,...,0]>0]

            loss_2d_pos_0 = criterion1(out_2d_0, gt_2d)
            loss_2d_pos_1 = criterion1(out_2d_1, gt_2d)
            loss_2d_pos_2 = criterion1(out_2d_2, gt_2d)

            loss_edge_0 = criterion1(edge_2d_0, gt_edges)
            loss_edge_1 = criterion1(edge_2d_1, gt_edges)
            loss_edge_2 = criterion1(edge_2d_2, gt_edges)
            # loss_heat_map = criterion1(heat_map_regress[labels>0].view(-1,2), gt_2d)
            loss_score = criterion2(out_2d_2.detach(), gt_2d, out_score)
            loss_edge_score = criterion2(edge_2d_2.detach(), gt_edges, edge_score)
            loss_2d_pos = 0.3*loss_2d_pos_0+0.5*loss_2d_pos_1+loss_2d_pos_2+ loss_score
            loss_edge = 0.3*loss_edge_0+0.5*loss_edge_1+loss_edge_2+loss_edge_score

            loss = loss_2d_pos + loss_edge
            loss.backward()

            if True:
                nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss_2d_pos.update(loss_2d_pos.item(),bz)
            # epoch_loss_heat_map.update(loss_heat_map.item(), bz)
            epoch_loss_score.update(loss_score.item(), bz)
            epoch_loss_edge.update(loss_edge.item(), bz)
            epoch_loss.update(loss.item(), bz)
            batch_time.update(time.time() - end)
            end = time.time()
            # add writer dict
            writer = writer_dict['writer']
            train_step = writer_dict['train_step']
            writer.add_scalar('train/total_loss', epoch_loss_2d_pos.avg, train_step)
            writer.add_scalar('train/score_loss', epoch_loss_score.avg, train_step)
            writer_dict['train_step'] = train_step + 1

            if _ % cfg.PRINT_FREQ == 0:
                bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                            '| Loss_kpts: {losskpts: .4f}| loss_score: {score:.4f}| loss_edges: {lossedges:.4f} '\
                            '| LR: {LR: .6f}| loss: {loss:.4f}' \
                    .format(batch=_ + 1, size=len(train_loader), data=data_time.val, bt=batch_time.avg,
                            ttl=bar.elapsed_td, eta=bar.eta_td, losskpts=epoch_loss_2d_pos.avg,
                            score=epoch_loss_score.avg, lossedges=epoch_loss_edge.avg, loss=epoch_loss.avg, LR=lr)
                bar.next()
        bar.finish()
        mAP,ap = eval_map(pose_generator,model_pos,test_loader,pred_json,best_json=cfg.best_json,target_json=cfg.target_json,flip_test=False)
        writer_map.write("{}\t{}\t{}\n".format(epoch,mAP,ap))
        writer_map.flush()
        valid_step = writer_dict['valid_step']
        writer.add_scalar('valid_mAP', mAP, valid_step)
        writer_dict['valid_step'] = valid_step + 1
        if best_map is None or best_map<mAP:
            best_map=mAP
            torch.save(model_pos.state_dict(), os.path.join(cfg.checkpoints, "best_checkpoint.pth"))

        model_pos.train()
        torch.set_grad_enabled(True)
        # save checkpoints
        save_dict = {
            'epoch': epoch,
            'model': cfg.model.type,
            'state_dict': model_pos.state_dict(),
            'map': best_map,
            'optimizer': optimizer.state_dict(),
            'train_step': writer_dict['train_step'],
            'valid_step': writer_dict['valid_step']
        }
        save_path = os.path.join(cfg.checkpoints,"{}.pth".format(epoch))
        torch.save(save_dict,save_path)
        root = '/home/tujun/projects/OPEC-Net'
        softlink = os.path.join(cfg.checkpoints,'checkpoint.pth')
        if os.path.exists(softlink):
            os.remove(softlink)
        os.symlink(os.path.join(root, save_path), softlink)
        