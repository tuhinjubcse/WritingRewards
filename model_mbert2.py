import argparse, os, shutil, torch, wandb, tqdm, json, numpy as np, torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from transformers import AdamW

def load_data(filename: str) -> List[Dict[str, Any]]:
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def create_dataloader(filename: str, batch_size: int) -> DataLoader:
    data = load_data(filename)
    
    examples = []
    for d in data:
        paragraph1, paragraph2 = "", "None"
        label_cls, label_reg = -1, -1
        if d["sample_type"] == "pairwise":
            label_cls = 0 if d["reference_preference"] == "1" else 1
            paragraph1 = d["paragraph1"]
            paragraph2 = d["paragraph2"]
        else:
            paragraph1 = d["paragraph"]
            label_reg = d["zscore"]

        rationale = "" if "rationale" not in d else d["rationale"]
            
        examples.append({
            "sample_type": d["sample_type"], 
            "paragraph1": paragraph1, 
            "paragraph2": paragraph2, 
            "label_cls": label_cls, 
            "label_reg": label_reg,
            "rationale": rationale
        })
    
    def collate_fn(batch):
        paragraphs1 = [x['paragraph1'] for x in batch]
        paragraphs2 = [x['paragraph2'] for x in batch]
        rationales = [x['rationale'] for x in batch]

        labels_cls = torch.LongTensor([x['label_cls'] for x in batch])
        labels_reg = torch.FloatTensor([x['label_reg'] for x in batch])
        
        return paragraphs1, paragraphs2, rationales, labels_cls, labels_reg
    
    return DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class MBertWritingReward(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if os.path.isdir(model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nlu = AutoModel.from_pretrained(model_name)
            state_dict = torch.load(os.path.join(model_name, 'heads.pth'), map_location=self.device, weights_only=True)
            
            hidden_size = self.nlu.config.hidden_size
            self.regression_head = self._create_regression_head(hidden_size)
            self.regression_head.load_state_dict(state_dict['regression_head'])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nlu = AutoModel.from_pretrained(model_name)
            
            hidden_size = self.nlu.config.hidden_size
            self.regression_head = self._create_regression_head(hidden_size)
        
        self.regression_scale = 10.0  # Scale factor for regression output
        self.to(self.device)
    
    def _create_regression_head(self, hidden_size):
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, paragraphs1, paragraphs2, rationales=None):
        N = len(paragraphs1)
        if rationales is None:
            rationales = [""] * N

        SEP_TOKEN_ID = self.tokenizer.sep_token_id
        
        # Process each pair of paragraphs
        all_input_ids, all_attention_masks, all_p1_ranges, all_p2_ranges = [], [], [], []
        
        for p1, p2, r in zip(paragraphs1, paragraphs2, rationales):
            p1_tokens = self.tokenizer(p1, add_special_tokens=False)['input_ids']
            p2_tokens = self.tokenizer(p2, add_special_tokens=False)['input_ids']

            r_tokens = []
            if r != "":
                r_tokens = self.tokenizer(r, add_special_tokens=False)['input_ids']

            
            input_ids = p1_tokens + [SEP_TOKEN_ID] + p2_tokens
            if len(r_tokens) > 0:
                input_ids += [SEP_TOKEN_ID] + r_tokens

            attention_mask = [1] * len(input_ids)
            
            p1_range = [0, len(p1_tokens)]
            p2_range = [len(p1_tokens) + 1, len(p1_tokens) + len(p2_tokens) + 1]
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_p1_ranges.append(p1_range)
            all_p2_ranges.append(p2_range)
        
        # Pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            padding_length = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
            padded_attention_masks.append(attention_mask + [0] * padding_length)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_input_ids).to(self.device)
        attention_mask = torch.tensor(padded_attention_masks).to(self.device)
        p1_ranges = torch.tensor(all_p1_ranges).to(self.device)
        p2_ranges = torch.tensor(all_p2_ranges).to(self.device)
        
        # Process through model
        outputs = self.nlu(input_ids=input_ids, attention_mask=attention_mask)

        p1_outputs, p2_outputs = [], []
        for i in range(N):
            p1_outputs.append(outputs.last_hidden_state[i, p1_ranges[i][0]:p1_ranges[i][1], :].mean(dim=0).unsqueeze(0))
            p2_outputs.append(outputs.last_hidden_state[i, p2_ranges[i][0]:p2_ranges[i][1], :].mean(dim=0).unsqueeze(0))

        p1_outputs = torch.cat(p1_outputs, dim=0)
        p2_outputs = torch.cat(p2_outputs, dim=0)

        reg_logits = self.regression_head(p1_outputs) * self.regression_scale  # Scale to 0-10 range
        cls_logits = torch.nn.functional.cosine_similarity(p1_outputs, p2_outputs, dim=-1)
        cls_logits = torch.clamp(cls_logits, min=1e-7, max=1)

        return cls_logits, reg_logits
    
    def predict_pair(self, paragraph1: str, paragraph2: str) -> str:
        self.eval()
        with torch.no_grad():
            cls_logits, _ = self([paragraph1], [paragraph2])
            pred = (cls_logits <= 0.5).long()
            return "1" if pred.item() == 0 else "2"
        
    def predict_regression(self, paragraph: str) -> float:
        self.eval()
        with torch.no_grad():
            _, reg_logits = self([paragraph], ["None"])
            return reg_logits.item()
    
    def evaluate(self, val_loader):
        self.eval()
        total_loss_cls, total_loss_reg = 0, 0
        total_acc, total_mse, total_mae = 0, 0, 0
        N_cls, N_reg = 0, 0

        all_reg_logits = []
        all_reg_labels = []
        with torch.no_grad():
            for paragraphs1, paragraphs2, rationales, labels_cls, labels_reg in val_loader:
                outputs = self(paragraphs1, paragraphs2, rationales)
                cls_logits, reg_logits = outputs
                
                for label_cls, label_reg, cls_logit, reg_logit in zip(labels_cls, labels_reg, cls_logits, reg_logits):
                    label_cls = label_cls.to(cls_logit.device)
                    label_reg = label_reg.to(reg_logit.device)
                    if label_cls != -1:
                        N_cls += 1
                        total_loss_cls += -torch.log(cls_logit) if label_cls == 0 else -torch.log(1 - cls_logit)

                        preds = (cls_logit <= 0.5).long()
                        total_acc += (preds == label_cls).sum().item()
                    else:
                        N_reg += 1
                        diff = (reg_logit - label_reg)
                        total_loss_reg += diff.pow(2).mean()
                        total_mse += diff.pow(2).mean()
                        total_mae += diff.abs().mean()
                        all_reg_logits.append(reg_logit.item())
                        all_reg_labels.append(label_reg.item())

        total_loss_cls = total_loss_cls / (N_cls + 1e-8)
        total_loss_reg = (total_loss_reg / (N_reg + 1e-8)) / 4.0
        total_loss = total_loss_cls + total_loss_reg
        total_acc = total_acc / (N_cls + 1e-8)
        total_mse = total_mse / (N_reg + 1e-8)
        total_mae = total_mae / (N_reg + 1e-8)

        val_corr = 0
        if N_reg > 0:
            val_corr = np.corrcoef(all_reg_logits, all_reg_labels)[0, 1]

        return {"loss_total": total_loss, "loss_cls": total_loss_cls, "loss_reg": total_loss_reg, "acc": total_acc, "mse": total_mse, "mae": total_mae, "N_cls": N_cls, "N_reg": N_reg, "val_corr": val_corr}

    def save_model(self, save_dir):
        """Save the model to a directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the base model and tokenizer
        self.nlu.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save the custom heads
        heads_state = {
            'regression_head': self.regression_head.state_dict()
        }
        torch.save(heads_state, os.path.join(save_dir, 'heads.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ModernBERT classifier')
    parser.add_argument('--model', type=str, default="answerdotai/ModernBERT-large")
    parser.add_argument('--train_fn', type=str, default="data/lamp_PR_train.json")
    parser.add_argument('--val_fn', type=str, default="data/lamp_PR_val.json")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_every', type=int, default=20)
    # parser.add_argument('--optim_every', type=int, default=1)        
    
    args = parser.parse_args()

    args.optim_every = 1
    if args.batch_size > 8:
        args.optim_every = args.batch_size // 8
        args.batch_size = (args.batch_size // args.optim_every)
        print(f'Batch size set to {args.batch_size} and optim_every set to {args.optim_every}')

    model_suffix = "PR" if "_PR_" in args.train_fn else "P" if "_P_" in args.train_fn else "R"

    wandb.init(
        project="writing-rewards-nlu",
        config={"train_fn": args.train_fn, "val_fn": args.val_fn, "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm, "epochs": args.epochs, "batch_size": args.batch_size, "eval_every": args.eval_every, "model": args.model, "optim_every": args.optim_every}
        )
    wandb.run.name = f'{model_suffix}-lr{args.learning_rate:.1e}-bs{args.batch_size}-oe{args.optim_every}'

    model = MBertWritingReward(args.model)
    train_loader = create_dataloader(args.train_fn, args.batch_size)
    val_loader = create_dataloader(args.val_fn, 100)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) // 10,  # Add warmup steps (10% of first epoch)
        num_training_steps=len(train_loader) * args.epochs
    )

    best_loss = 100.0 # nothing above 0.65 needs to be saved
    for epoch in range(args.epochs):
        for batch_idx, (paragraphs1, paragraphs2, rationales, labels_cls, labels_reg) in enumerate(tqdm.tqdm(train_loader)):
            outputs = model(paragraphs1, paragraphs2, rationales)
            
            cls_logits, reg_logits = outputs
            labels_cls = labels_cls.to(cls_logits.device)
            labels_reg = labels_reg.to(reg_logits.device)

            n_reg, n_cls = 0, 0
            loss_cls, loss_reg = 0, 0
            for label_cls, label_reg, cls_logit, reg_logit in zip(labels_cls, labels_reg, cls_logits, reg_logits):
                if label_cls != -1: # it's classification
                    n_cls += 1
                    loss_item = -torch.log(cls_logit) if label_cls == 0 else -torch.log(1 - cls_logit)
                    loss_cls += loss_item
                else:
                    n_reg += 1
                    loss_item = (reg_logit - label_reg).pow(2)
                    loss_reg += loss_item

            loss_cls = loss_cls / (n_cls + 1e-8)
            loss_reg = (loss_reg / (n_reg + 1e-8)) / 4.0 # just to scale it to the cls loss

            # Combined loss
            if n_reg > 0 and n_cls > 0:
                loss = loss_cls + loss_reg
            elif n_reg > 0:
                loss = loss_reg
            else:
                loss = loss_cls

            # Scale loss for gradient accumulation
            loss = loss / args.optim_every
            
            train_log = {'train/loss': loss.item(), 'train/n_reg': n_reg, 'train/n_cls': n_cls}
            if n_reg > 0:
                train_log["train/loss_reg"] = loss_reg.item()
            if n_cls > 0:
                train_log["train/loss_cls"] = loss_cls.item()
            wandb.log(train_log)
            
            loss.backward()

            # Only optimize every optim_every steps
            if (batch_idx + 1) % args.optim_every == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if epoch >= 1 and (batch_idx + 1) % args.eval_every == 0:
                val_metrics = model.evaluate(val_loader)
                wandb.log({'val/loss': val_metrics["loss_total"], 'val/loss_cls': val_metrics["loss_cls"], 'val/loss_reg': val_metrics["loss_reg"], 'val/accuracy': val_metrics["acc"], 'val/mse': val_metrics["mse"], 'val/mae': val_metrics["mae"], 'val/corr': val_metrics["val_corr"], 'val/N_cls': val_metrics["N_cls"], 'val/N_reg': val_metrics["N_reg"], "val/best_loss": best_loss})

                print(f'Epoch {epoch+1}/{args.epochs}, Val Loss: {val_metrics["loss_total"]:.4f} (cls: {val_metrics["loss_cls"]:.4f} + reg: {val_metrics["loss_reg"]:.4f}) acc: {val_metrics["acc"]:.4f}; mse: {val_metrics["mse"]:.4f}; mae: {val_metrics["mae"]:.4f}; corr: {val_metrics["val_corr"]:.4f})')
                if val_metrics["loss_total"] < min(best_loss, 0.60):
                    # Delete previous best model folder if it exists
                    if hasattr(model, 'best_model_dir') and os.path.exists(model.best_model_dir):
                        shutil.rmtree(model.best_model_dir)
                    
                    best_loss = val_metrics["loss_total"]
                    save_dir = f'models/mbert2-large-{model_suffix}-loss{best_loss:.3f}'
                    print(f'\033[94mSaving model to {save_dir}\033[0m')
                    model.best_model_dir = save_dir
                    model.save_model(save_dir)  # Use the new save_model method
                model.train()

    wandb.finish()
