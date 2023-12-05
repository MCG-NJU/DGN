from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

def eval_results(preds,target_json):
    gt_file = target_json
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(preds)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__=='__main__':

    gt = '/home/tujun/projects/deep-high-resolution-net.pytorch/data/posetrack/annotations/val2017_expand.json'
    pred = '/home/tujun/Assist_Codes/posetrack2017/keypoints_val2017_results_posetrack_hrnetw48_256x192_detbbox47.json'
    

    eval_results(pred, gt)