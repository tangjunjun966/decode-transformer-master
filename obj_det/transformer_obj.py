

import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDec(nn.Module):
    '''
    d_model=512, 使用多少维度表示，实际为编码输出表达维度
    nhead=8, 有多少个头
    num_queries=100, 目标查询数量，可学习query
    num_decoder_layers=6, 解码循环层数
    dim_feedforward=2048, 类似FFN的2个nn.Linear变化
    dropout=0.1,
    activation="relu",
    normalize_before=False,解码结构使用2种方式，默认False使用post解码结构
    output_intermediate_dec=False, 若为True保存中间层解码结果(即：每个解码层结果保存)，若False只保存最后一次结果，训练为True，推理为False
    num_classes: num_classes数量与数据格式有关，若类别id=1表示第一类，则num_classes=实际类别数+1，若id=0表示第一个，则num_classes=实际类别数

    额外说明，coco类别id是1开始的，假如有三个类,名称为[dog,cat,pig],batch=2,那么参数num_classes=4，表示3个类+1个背景，
    模型输出src_logits=[2,100,5]会多出一个预测，target_classes设置为[2,100]，其值为4(该值就是背景，而有类别值为1、2、3),
    那么target_classes中没有值为0，我理解模型不对0类做任何操作，是个无效值，模型只对1、2、3、4进行loss计算，然4为背景会比较多，
    作者使用权重0.1避免其背景过度影响。

    forward return: 返回字典，包含{
    'pred_logits':[],  # 为列表，格式为[b,100,num_classes+2]
    'pred_boxes':[],  # 为列表，格式为[b,100,4]
    'aux_outputs'[{},...] # 为列表，元素为字典，每个字典为{'pred_logits':[],'pred_boxes':[]}，格式与上相同

    }

    '''

    def __init__(self, d_model=512, nhead=8, num_queries=100, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, output_intermediate_dec=False, num_classes=1):
        super().__init__()

        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)  # 与编码输出表达维度一致
        self.output_intermediate_dec = output_intermediate_dec


        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=output_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead



        # 设置head头提取

        self.num_classes=num_classes
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)



    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_feature(self, x, pos_embed, mask=None):
        '''
        假设batch=2，编码后的宽高为17、23
        x: 编码后的输入，变为[17*23,2,256]
        pos_embed:位置编码，使用[17*23,1,256]
        mask:为图像掩码，nn.MultiheadAttention函数使用，作为memory_key_padding_mask的参数，默认为None，
        被图像覆盖为0，否则为1，格式为[2,17,23]。个人感觉pad填充位置为1，就忽略注意，那里是没意义关注的。
        '''

        # flatten NxCxHxW to HWxNxC
        hw,bs,c=x.shape  # 编码后的输入，格式为h*w,b,c

        pos_embed = pos_embed.repeat(1, bs, 1)  # 位置变成[17*23,2,256]

        # self.query_embed.weights 是可训练的
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # 可学习参数从[100,256]变成[100,2,256]
        if mask is not None:
            mask = mask.flatten(1)  # mask从[2,17,23]变成[2,17*23],该mask被图像覆盖为0
        tgt = torch.zeros_like(query_embed)  # 生成0张量[100,2,256]

        # tgt为生成0张量[100,2,256]，memory为编码输出[17*23,2,256]，pos为位置编码不变，query_embed为可学习查询参数，在解码开始学习
        hs = self.decoder(tgt, x, memory_key_padding_mask=mask,  pos=pos_embed, query_pos=query_embed)  # [6,100,2,256]
        return hs.transpose(1, 2)

    def forward(self,x,pos_embed,mask=None):
        '''
        假设batch=2，编码后的宽高为17、23
        x: 编码后的输入，变为[17*23,2,256]
        pos_embed:位置编码，使用[17*23,1,256]
        mask:为图像掩码，nn.MultiheadAttention函数使用，作为memory_key_padding_mask的参数，默认为None，
        被图像覆盖为0，否则为1，格式为[2,17,23]。个人感觉pad填充位置为1，就忽略注意，那里是没意义关注的。
        '''

        feature_dec = self.forward_feature( x, pos_embed, mask=mask)

        # 推理输出类别与box
        outputs_class = self.class_embed(feature_dec)  # [6,2,100,92] 92=cls_num+背景+置信度，在如三个类[person，cat，dog]，则为3+1+1
        outputs_coord = self.bbox_embed(feature_dec).sigmoid()  # [6,2,100,4]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}  # 都是取最后一个，变成[2,100,92]和[2,100,4]
        # 上面内容训练与推理适用
        if self.output_intermediate_dec:
            # out['aux_outputs']将前面5个值匹配对应
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt  # output初始化为0，[100,2,256]

        intermediate = []

        for layer in self.layers:  # 循环遍历5次，结构都一样
            output = layer(output, memory, tgt_mask=tgt_mask,   #output初始化为0[100,2,256] memory为编码输入特征[551,2,256]，tgt_mask为None
                           memory_mask=memory_mask,  # memory为None
                           tgt_key_padding_mask=tgt_key_padding_mask,  # 为None
                           memory_key_padding_mask=memory_key_padding_mask,  # 和编码一样[2,551]
                           pos=pos, query_pos=query_pos)  # pos为位置编码，query_pos为可学习query[100,2,256]
            if self.return_intermediate:
                intermediate.append(self.norm(output))  # 每次解码输出结果[100,2,256]
        # output 输出仍为[100,2,256]
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)  # 将多intermediate保存[100,2,256]做拼接，输出为[6,100,2,256]
        return output.unsqueeze(0)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)  # tgt为0，query_pos为可学习参数，随机初始化的
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  # q k加了位置分开，而tgt是没加位置信息
        tgt = tgt + self.dropout1(tgt2)  # 类似残差连接
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),  # query为自注意力后tgt+query [100,2,256]
                                   key=self.with_pos_embed(memory, pos),  # key为编码后的值加位置[391,2,256]
                                   value=memory, attn_mask=memory_mask,  # value为编码后的值[391,2,256]
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)  # tgt再次加上获得tgt2值
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def train_decode_demo():

    M = TransformerDec(d_model=256, output_intermediate_dec=True, num_classes=4)

    src = torch.rand((391, 2, 256))

    pos_embed = torch.ones((391, 1, 256))

    res = M(src, pos_embed)






#*****************************推理代码************************#

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)  # 从输出[..., :-1]找最大值，最后一个值不管，而对应最大值顺势为score

        # convert to [x0, y0, x1, y1] format
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)



def predect_demo():

    '''
    这是一个推理模型，我使用batch=2进行推理验证的demo
    当然，这个没有载入权重，仅作为推理演示
    '''

    batch = 2
    src = torch.rand((391, batch, 256))
    pos_embed = torch.ones((391, 1, 256))



    M = TransformerDec(d_model=256, output_intermediate_dec=True, num_classes=4).cuda()



    res = M(src.cuda(), pos_embed.cuda())
    h, w = 420, 640  # 原始图像尺寸，为后处理恢复box使用
    orig_target_sizes = torch.tensor([[h, w], [h, w]]).cuda()  # batch为2，需要2个高宽
    result = PostProcess()(res, orig_target_sizes)  # orig_target_sizes原始图像尺寸，未resize尺寸

    # print(res)
    res_index, res_score, res_lable, res_bbox = [], [], [], []
    min_score = 0.5  # thr的阈值，默认为0.9


    for b in range(batch):
        for i in range(0, 100):
            res_tmp = result[b]['scores']
            if float(res_tmp[i]) > min_score:
                res_score.append(float(res_tmp[i]))

                res_lable.append(int(result[b]['labels'][i].cpu().numpy()))
                res_bbox.append(result[b]['boxes'][i].cpu().numpy().tolist())
    print("result: ", res_score, res_lable, res_bbox)






if __name__ == '__main__':
    train_decode_demo()
    # predect_demo()



