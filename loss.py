import torch

import torch.nn as nn

import torch.nn.functional as F



class YOLO_loss(nn.Module):
    def __init__(self, f_size=7, n_bb=2, n_cls=20, coeff_coord=5.0, coeff_noobj=0.5, _gpu=True):

        super(YOLO_loss, self).__init__()
        self.S, self.B, self.C = f_size, n_bb, n_cls
        self.coeff_coord = coeff_coord
        self.coeff_noobj = coeff_noobj
        self.gpu = _gpu


    def iou(self, bbox1, bbox2):

        """
            Args:

                bbox1 & bbox2 (Tensor): bounding boxe with shape [N, 4] & [M, 4] respectively

            Returns:

                (Tensor) IoU, with shape [N, M]

            =================================================================================    

            IoU: Intersection of Union

             + Intersection/Union

            Union = bbox1's area + bbox2's area - intersection

            for bboxes

                area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

                which equals to (x2-x1) * (y2-y1)


            for Intersection        

                Compute the left-top & right-bottom coordinates of the intersection.\

                   + get max x1, y1 values for the left-top one

                   + get min x2, y2 values for the right-bottom one

            ==================================================================================

        """
        bbox1_x1y1 = bbox1[:, :2]/float(self.S) - 0.5 * bbox1[:, 2:]
        bbox1_x2y2 = bbox1[:, :2]/float(self.S) + 0.5 * bbox1[:, 2:]
        bbox2_x1y1 = bbox2[:, :2]/float(self.S) - 0.5 * bbox2[:, 2:]
        bbox2_x2y2 = bbox2[:, :2]/float(self.S) + 0.5 * bbox2[:, 2:]

        box1 = torch.cat((bbox1_x1y1.view(-1,2), bbox1_x2y2.view(-1, 2)), dim=1) # [N,4], 4=[x1,y1,x2,y2]
        box2 = torch.cat((bbox2_x1y1.view(-1,2), bbox2_x2y2.view(-1, 2)), dim=1) # [M,4], 4=[x1,y1,x2,y2]


        N = box1.size(0)
        M = box2.size(0)

        coord_lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            box2[:, :2].unsqueeze(0).expand(N, M, 2)
        )
        coord_rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            box2[:, 2:].unsqueeze(0).expand(N, M, 2)
        )

        # Compute Intersection area
        WaH = coord_rb - coord_lt
        WaH[WaH<0] = 0
        intersection = WaH[:, :, 0] * WaH[:, :, 1]

        # Compute Union area
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        box1_area = box1_area.unsqueeze(1).expand_as(intersection)
        box2_area = box2_area.unsqueeze(0).expand_as(intersection)

        union = box1_area + box2_area - intersection
        iou = intersection/union
        return iou

        

    def forward(self, pred, target):

        """

        Args:

            pred: (Tensor) prediction with shape [Batch, S, S, 5*B + C]

            target: (Tensor) target with shape   [Batch, S, S, 5*B + C]

        Returns:

            (Tensor) loss with shape [1,]

        """

        if self.gpu:
            BoolTensor = torch.cuda.BoolTensor
            FloatTensor = torch.cuda.FloatTensor
        else:
            BoolTensor = torch.BoolTensor
            FloatTensor = torch.FloatTensor

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        pred, target = pred, target

        

        batch_size = pred.size(0)
        target = target.view(batch_size, -1, N)
        pred   = pred.view(batch_size, -1, N)


        yeobj_mask = target[:, :, 4] > 0  # object 있는 cell
        noobj_mask = target[:, :, 4] == 0 # object 없는 cell


        # shape change from [Batch, S, S] => [Batch, S, S, N]
        yeobj_mask = yeobj_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target) 

        """

            Cells which have object

              + change shape from [Batch, S, S, N] => [#yeobj, N]

              + bbox_pred:  get [:#yeobj, :5*B] and reshape to [#yeobj*5, 5]

              + class_pred: get [:#yeobj, 5*B:]

        """

        yeobj_pred = torch.masked_select(pred, yeobj_mask).view(-1, N)   # [Batch, S, S, N] => [#yeobj, N]
        bbox_pred  = yeobj_pred[:, :5*B].contiguous().view(-1, 5)        # [#yeobj, N]      => [#yeobj * B, 5(x, y, w, h, confidence)]
        class_pred = yeobj_pred[:, 5*B:]                                 # [#yeobj, N]      => [#yeobj, C]


        yeobj_targ = torch.masked_select(target, yeobj_mask).view(-1, N)
        bbox_targ  = yeobj_targ[:, :5*B].contiguous().view(-1, 5)
        class_targ = yeobj_targ[:, 5*B:]


        """
            Cells which do not have object
              + change shape from [Batch, S, S, N] => [#noobj, N]
        """

        noobj_pred = torch.masked_select(pred, noobj_mask).view(-1, N)
        noobj_targ = torch.masked_select(target, noobj_mask).view(-1, N)

        
        """

            No object cells confidences mask

             ==> apply to noobj_pred & targ to get both pred & targ confidences

                 shape [#noobj, N] => [#noobj, 2(conf1, conf2)]

            

            & Calculate Loss for noobj

        """


        noobj_conf_mask = BoolTensor(noobj_pred.size())
        noobj_conf_mask.zero_()
        noobj_conf_mask[:, 4].fill_(1), noobj_conf_mask[:, 9].fill_(1)

        noobj_pred_conf = noobj_pred[noobj_conf_mask].clone().view(-1, 2)
        noobj_targ_conf = torch.zeros_like(noobj_pred_conf)

        yeobj_response_mask     = BoolTensor(bbox_targ.size())
        bbox_targ_iou           = FloatTensor(bbox_targ.size())
        yeobj_response_mask.zero_()
        bbox_targ_iou.zero_()

        for i in range(0, bbox_targ.size(0), B):

            """
                pred_: all the predicted bboxes at i-th cell    | [B, 5(x, y, w, h, conf)] 
                targ_: all the target bboxes at i-th cell 
            """

            pred_ = bbox_pred[i:i+B]
            targ_ = bbox_targ[i:i+B]

            iou = self.iou(pred_[:, :4], targ_[:, :4])

            max_iou, max_index = iou.max(0)
            max_index = max_index.data
            yeobj_response_mask[i+max_index] = 1
            bbox_targ[i+max_index, 4] = max_iou.data



        bbox_pred_response = bbox_pred[yeobj_response_mask].view(-1, 5)
        bbox_targ_response = bbox_targ[yeobj_response_mask].view(-1, 5)

        loss_loc   = F.mse_loss(bbox_pred_response[:, :2], bbox_targ_response[:, :2], reduction='sum') + \
                     F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_targ_response[:, 2:4]), reduction='sum')
        loss_yeobj   = F.mse_loss(bbox_pred_response[:, 4], bbox_targ_response[:, 4], reduction='sum')
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_targ_conf, reduction="sum")
        loss_cls   = F.mse_loss(class_pred, class_targ, reduction='sum')

        total_loss = self.coeff_coord * loss_loc + loss_yeobj + self.coeff_noobj * loss_noobj + loss_cls
        total_loss /= float(batch_size)
        return total_loss
