import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", model_max_length=512)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskBertForRoles(nn.Module):
    def __init__(self, num_main_roles, num_fine_grained_roles):
        super(MultiTaskBertForRoles, self).__init__()
        self.bert = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        self.dropout = nn.Dropout(p=0.3)
        self.main_role_classifier = nn.Linear(self.bert.config.hidden_size * 3, num_main_roles)  
        self.fine_grained_classifier = nn.Linear(self.bert.config.hidden_size * 3, num_fine_grained_roles)  

        self.attention_weights = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, entity_spans=None, main_labels=None, fine_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output = outputs.last_hidden_state[:, 0, :] 
        
        if entity_spans is not None:
            entity_embeds = torch.stack([
                outputs.last_hidden_state[b, start:end].mean(dim=0) 
                for b, (start, end) in enumerate(entity_spans)
            ])
        else:
            entity_embeds = cls_output 
        
        pooled_output = torch.cat([cls_output, entity_embeds], dim=1)
        pooled_output = self.dropout(pooled_output)
        
        attention_scores = F.softmax(self.attention_weights(outputs.last_hidden_state), dim=1) 
        context_vector = torch.sum(outputs.last_hidden_state * attention_scores, dim=1) 
        
        pooled_output = torch.cat([pooled_output, context_vector], dim=1)
        
        main_logits = self.main_role_classifier(pooled_output)
        fine_logits = self.fine_grained_classifier(pooled_output)
        
        loss = None
        if main_labels is not None and fine_labels is not None:
            if main_labels.ndim > 1:
                main_labels = torch.argmax(main_labels, dim=1)

            main_loss_fn = nn.CrossEntropyLoss()
            main_loss = main_loss_fn(main_logits, main_labels)

            fine_loss_fn = FocalLoss(alpha=1, gamma=2)
            fine_loss = fine_loss_fn(fine_logits, fine_labels)

            loss = 0.10 * main_loss + 0.90 * fine_loss

        return loss, main_logits, torch.sigmoid(fine_logits)
