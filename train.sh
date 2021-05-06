CUDA_VISIBLE_DEVICES=0,1,3,4 python ./tools/train_alpha_pose_gcn.py --indir ../crowdpose/images/ --trainBatch $1 --validBatch 128 --dataset 'coco' --config $2
# test batch = 24 * 4 = 96