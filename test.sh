CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/test_alpha_pose_gcn.py --indir ../crowdpose/images/  --load_dirs $1 --validBatch 84  --dataset 'coco' --config $2
