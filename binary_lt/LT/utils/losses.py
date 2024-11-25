import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, input_size: int, teacher_size: int, weighted_distillation: bool, weight= None,args = None):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.input_size = input_size
        self.teacher_size = teacher_size
        self.weighted_distillation = weighted_distillation
        self.weight = weight
        self.args = args


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            #print("This is used when there is a teacher model with a distillationhead")
            #outputs, outputs_kd = outputs[0], outputs[1]
            outputs, outputs_kd = outputs, outputs
        try:
            base_loss = self.base_criterion(outputs, labels)
        except:
            labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes = len(self.weight)).cuda()
            base_loss = self.base_criterion(outputs, labels)


        if self.distillation_type == 'none':
            if not self.weighted_distillation:
                token_2_loss = SoftTargetCrossEntropy()(outputs_kd, labels)
            else:
                token_2_loss = torch.nn.CrossEntropyLoss(weight = self.weight)(outputs_kd, labels)
            loss = base_loss * (1 - self.alpha) + token_2_loss * self.alpha
            return loss, base_loss * (1 - self.alpha), token_2_loss * self.alpha 

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher

          
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            pred_t = F.log_softmax(teacher_outputs / T, dim=1)
            if self.weighted_distillation:
                pred_t = pred_t * self.weight
                pred_t = pred_t / pred_t.sum(1)[:, None]

            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                pred_t,
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':

            distillation_targets = teacher_outputs.argmax(dim=1).cuda() #[256]
            if self.args.map_targets:
                distillation_targets = torch.Tensor(np.array(self.args.class_map)[distillation_targets.detach().cpu()]).type(torch.LongTensor).cuda()


            if self.weighted_distillation:
                #print("Weighted Distillation")
                distillation_loss = F.cross_entropy(outputs_kd, distillation_targets, weight = self.weight)
            else:
                distillation_loss = F.cross_entropy(outputs_kd, distillation_targets)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss, base_loss * (1 - self.alpha), distillation_loss * self.alpha
    

class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = ( features.shape[0] - self.K ) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth 
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
