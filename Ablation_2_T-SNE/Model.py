import os
import torch
import numpy as np
from torch import nn
from transformers import (
    BertModel,
    BertPreTrainedModel,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from Function_For_KM import adjust_tensor_by_proportion

class AbstractModule(nn.Module):
    def __init__(self):
        super(AbstractModule, self).__init__()
        pass

    def forward(self, x):
        return x        
        
class KM(BertPreTrainedModel):
    def __init__(self, config, input_dim, encoding_dim, CC, strech_proportion):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.CC = CC
        
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
        self.strech_proportion = strech_proportion

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        if_change: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        
        # 动态调整特征（保持原有逻辑）
        if self.training and if_change is not None:
            if_change = if_change.to(pooled_output.device)
            mask = if_change.bool()
            
            if mask.any():
                CC_device = self.CC.to(pooled_output.device)
                # 批量处理需要调整的样本
                adjusted = adjust_tensor_by_proportion(
                    CC_device, 
                    pooled_output[mask], 
                    self.strech_proportion
                )
                # 创建新张量避免原地操作
                adjusted_output = pooled_output.clone()
                adjusted_output[mask] = adjusted
                pooled_output = adjusted_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 损失计算保持不变
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )