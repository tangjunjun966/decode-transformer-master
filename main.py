
import torch
from obj_det.transformer_obj import TransformerDec
from losses.matcher import HungarianMatcher
from losses.loss import SetCriterion

if __name__ == '__main__':


    Model = TransformerDec(d_model=256, output_intermediate_dec=True, num_classes=4)

    num_classes = 4   #  类别+1
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)  # 二分匹配不同任务分配的权重
    losses = ['labels', 'boxes', 'cardinality']  # 计算loss的任务
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}  # 为dert最后一个设置权重
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)

    # 下面使用iter，我构造了虚拟模型编码数据与数据加载标签数据
    src = torch.rand((391, 2, 256))
    pos_embed = torch.ones((391, 1, 256))

    # 创造真实target数据
    target1 = {'boxes':torch.rand((5,4)),'labels':torch.tensor([1,3,2,1,2])}
    target2 = {'boxes': torch.rand((3, 4)), 'labels': torch.tensor([1, 1, 2])}
    target = [target1, target2]

    res = Model(src, pos_embed)
    losses = criterion(res, target)
    print(losses)













