CUDA_VISIBLE_DEVICES=2,3,0,1 python ./tools/train_alpha_pose_gcn.py --indir ../crowdpose/images/ --nEpochs 25 --trainBatch $1 --validBatch 64 --dataset 'coco' --config $2
# test batch = 24 * 4 = 96