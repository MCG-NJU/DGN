CUDA_VISIBLE_DEVICES=4,5,6,7 python ./tools/test_alpha_pose_gcn.py --indir ../crowdpose/images/  --validBatch 256 --dataset 'CrowdPose' --config $1
