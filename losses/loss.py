
import torch.nn as nn
import torch
import torch.nn.functional as F
# from utils.losses.losses.box_ops import generalized_box_iou,box_cxcywh_to_xyxy
from losses import box_ops
from losses.misc import (NestedTensor, nested_tensor_from_tensor_list,
                         accuracy, get_world_size, interpolate,
                         is_dist_avail_and_initialized)




class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses_task = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 只获得类别预测结果，[2,100,5]

        idx = self._get_src_permutation_idx(indices)  # idx为tuple(tensor([0,1]),tensor([67,79]))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # 获得对应gt的类别,为[1,2]
        """
        # 说明，coco类别id是1开始的，假如有三个类,名称为[dog,cat,pig],batch=2,那么参数num_classes=4，表示3个类+1个背景，
        模型输出src_logits=[2,100,5]会多出一个预测，target_classes设置为[2,100]，其值为4(该值就是背景，而有类别值为1、2、3),
        那么target_classes中没有值为0，我理解模型不对0类做任何操作，是个无效值，模型只对1、2、3、4进行loss计算，然4为背景会比较多，
        作者使用权重0.1避免其背景过度影响。

        """

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)

        # 该部分就是论文所说使用某种方式将gt也变成100的方式，赋值标签id，第一类的标签为1，以此类推
        target_classes[idx] = target_classes_o  # 将对应idx赋值，即[0,67]位置为1，[1，79]位置为2，其它赋值任为4
        # src_logits.transpose(1, 2) 变为[2,5,100],而target_classes变为[2,100]
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        loss_ce = loss_ce*self.weight_dict['loss_ce'] if 'loss_ce' in self.weight_dict.keys() else loss_ce
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]


        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']  # 获得类别预测[2,100,5]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)  # 获得每个图box数量为一维张量[1,1]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # 最后一个值为4表示没有值
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)  # 每张图对应预测pre=100没有目标判断
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())  # 数量做了L1 loss
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)  # 这里与labels一致
        src_boxes = outputs['pred_boxes'][idx]  # outputs['pred_boxes']为[2,100,4],通过idx索引获得对应预测box，[2,4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # 获得对应gt box，[2,4]
        # 这里说明下gt box就是对应中心点与宽高(与yolov5数据txt一样)，并与预测box直接求loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')  # 做了L1 LOSS，输出维度[2,4]

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes  # 求和并除以box总数

        if 'loss_bbox' in self.weight_dict.keys():
            losses['loss_bbox'] = losses['loss_bbox'] * self.weight_dict['loss_bbox']

        # 这一步是giou loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes  # 求和并除以box总数
        if 'loss_giou' in self.weight_dict.keys():
            losses['loss_giou'] = losses['loss_giou'] * self.weight_dict['loss_giou']

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  # 获得batch，即图像索引
        src_idx = torch.cat([src for (src, _) in indices])  # 按顺序获得预测对应索引
        return batch_idx, src_idx  # 输出图像索引与预测对应索引

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)  # 通过名称获得不同loss函数，但输入值都是一样的

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # outputs_without_aux获得pred_logits[2,100,5]和pred_boxes[2,100,4]
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)  # 获得所有gt目标数量
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses_task:  # labels,boxes,cardinality
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:  # 这里得到其它曾也向上面那样在做一次loss
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses_task:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses_sum = sum([v for k, v in losses.items() if 'error' not in k])

        return losses_sum


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes




if __name__ == '__main__':

    import torch

    from losses.matcher import HungarianMatcher

    num_classes = 4   #  类别+1
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)  # 二分匹配不同任务分配的权重
    losses = ['labels', 'boxes', 'cardinality']  # 计算loss的任务
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}  # 为dert最后一个设置权重
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)


    # 创造真实target数据
    target1 = {'boxes':torch.rand((5,4)),'labels':torch.tensor([1,3,2,1,2])}
    target2 = {'boxes': torch.rand((3, 4)), 'labels': torch.tensor([1, 1, 2])}
    target = [target1, target2]

    pred_logits=torch.rand((2, 100, 5))
    pred_boxes = torch.rand((2, 100, 4))
    res={'pred_logits':pred_logits,'pred_boxes':pred_boxes}


    losses = criterion(res, target)
    print(losses)





