# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
import cv2
import os

# def visualize_masks(masks, output_dir='masks_new'):
#     """
#     Visualize binary mask tensor with shape (N, H, W) and save them as PNG images in the output directory.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     masks = masks.flatten(0, 1)
#     print('masks shape:', masks.shape)
#     n, h, w = masks.shape

#     for i in range(n):
#         mask = masks[i].cpu().numpy()
#         mask = (mask * 255).astype('uint8')
#         # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         filename = os.path.join(output_dir, f'mask_{i}.png')
#         cv2.imwrite(filename, mask)

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return masks
    
    n = masks.shape[0]
    masks = masks.flatten(0, 1)
    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if len(x) * len(y) == 0:
            continue
        
        masks[index, torch.min(y):torch.max(y)+1, torch.min(x):torch.max(x)+1] = 1.0
    
    masks = masks.view(n, -1, masks.shape[-2], masks.shape[-1])
    return masks

def masks_to_boxes_new(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return masks
    
    n, _, h, w = masks.shape
    masks = masks.flatten(0, 1)
    y = torch.arange(0, h, dtype=torch.float).to(masks.device)
    x = torch.arange(0, w, dtype=torch.float).to(masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0] + 1
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0] + 1
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    boxes = torch.stack([x_min, y_min, x_max, y_max], 1)
    # print('boxes shape:', boxes.shape)

    mem_mask = torch.zeros_like(masks)

    hMask = torch.logical_or(torch.arange(h).unsqueeze(0).to(boxes)<boxes[:, 1, None], torch.arange(h).unsqueeze(0).to(boxes)>=boxes[:, 3, None])
    wMask = torch.logical_or(torch.arange(w).unsqueeze(0).to(boxes)<boxes[:, 0, None], torch.arange(w).unsqueeze(0).to(boxes)>=boxes[:, 2, None])
    
    mem_mask = torch.logical_or(hMask.unsqueeze(2), wMask.unsqueeze(1)).float()
    # print('mem mask shape:', mem_mask.shape)
    mem_mask = 1.0 - mem_mask.view(n, -1, masks.shape[-2], masks.shape[-1])
    return mem_mask

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_dice_loss_nosig(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

batch_dice_loss_jit_nosig = torch.jit.script(
    batch_dice_loss_nosig
)  # type: torch.jit.ScriptModule

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    return loss / hw

def batch_sigmoid_ce_loss_nosig(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    #print('loss max no sig:', loss.max())
    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

batch_sigmoid_ce_loss_jit_nosig = torch.jit.script(
    batch_sigmoid_ce_loss_nosig
)  # type: torch.jit.ScriptModule

class VideoHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            is_ytvis = (out_mask.shape[1] == 3) # change here

            if is_ytvis:
                # out_mask_c = masks_to_boxes((out_mask.sigmoid() > 0.5).clone().float()).float()
                out_mask = masks_to_boxes_new((out_mask.sigmoid() > 0.5).float()).float() # ori match
                # visualize_masks(out_mask, 'box_mask_convert')
                
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]
            
            if is_ytvis:
                tgt_mask = masks_to_boxes(tgt_mask).float() # ori match, change here will also influnce criterion

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                
                # Compute the dice loss betwen masks
                if not is_ytvis:
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                else:
                    cost_dice_nosig = batch_dice_loss_jit_nosig(out_mask, tgt_mask)

            # Final cost matrix
            if not is_ytvis:
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
            else:
                C = (
                    self.cost_class * cost_class
                    + self.cost_dice * cost_dice_nosig
                )

            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
