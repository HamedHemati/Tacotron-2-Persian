import os
import sys
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tac2persian.models.tacotron2 import Tacotron2
from tac2persian.utils.generic import load_config
from tac2persian.utils.path_manager import PathManager
from tac2persian.dataset import get_tacotron2_dataloader
from tac2persian.utils.g2p.char_list import char_list
from tac2persian.models.modules_tacotron2 import pad_mask
from tac2persian.utils.plot import plot_attention, plot_spectrogram
from tac2persian.utils.display import stream


class TacotronTrainer():
    def __init__(self, config, path_manager, model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.path_manager = path_manager
        self.model = model.to(self.device)
        self.writer = SummaryWriter(log_dir=self.path_manager.logs_path)
        self._init_criterion_optimizer()
        self._init_dataloaders()

    def _init_criterion_optimizer(self):
        reduction = "none" if self.config["use_weighted_masking"] else "mean"
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction=reduction, 
                                                  pos_weight=torch.tensor(self.config["bce_pos_weight"]))
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        
    
    def _init_dataloaders(self):
        # Load train loader
        self.train_loader, attn_example, self.speaker_to_id = get_tacotron2_dataloader(self.config,
                                                                                       eval=False, 
                                                                                       binned_sampler=True)
        
        # Save speaker_to_id for inference
        with open(os.path.join(self.path_manager.output_path, "speakers.yml"), "w") as speakers_file:
            yaml.dump(self.speaker_to_id, speakers_file)

        # Set example
        _, self.example_id, self.example_meta = attn_example
        
        # Load eval loader
        self.eval_loader, _, _ = get_tacotron2_dataloader(self.config,
                                                          eval=True, 
                                                          binned_sampler=True)

    def _compute_loss(self, 
                      outputs, 
                      postnet_outputs, 
                      mel, 
                      mel_len, 
                      stop_values, 
                      stop_labels):
        # Mel-spec loss
        l1_loss = self.l1_criterion(postnet_outputs, mel) + self.l1_criterion(outputs, mel)
        mse_loss = self.mse_criterion(postnet_outputs, mel) + self.mse_criterion(outputs, mel)
       
        # Stop loss
        bce_loss = self.bce_criterion(stop_values, stop_labels)
        
        # Compute weight masks and apply reduction
        if self.config["use_weighted_masking"]:
            r = self.model.get_reduction_factor() 
            mel_len_ = mel_len.cpu().numpy()
            masks = pad_mask(mel_len_, r).unsqueeze(-1).to(self.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(mel.size(0) * mel.size(2))
            logit_weights = weights.div(mel.size(0))
            
            # Apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = bce_loss.mul(logit_weights.squeeze(-1)).masked_select(masks.squeeze(-1)).sum()
        
        # Compute total loss
        loss = l1_loss + mse_loss + bce_loss

        return loss, l1_loss, mse_loss, bce_loss

    def train(self):
        for epoch in range(self.config["epochs"]):
            self._train_epoch(epoch)
            self._eval_epoch(epoch)
            
    def _train_epoch(self, epoch):
        self.model.train()
        device = self.device
        
        running_loss = 0
        total_iters = len(self.train_loader)
        # Train for one epoch
        for itr, (batch_items) in enumerate(self.train_loader, 1):
            item_id, inp_chars, chars_len, mel, mel_len, speaker_id, stop_labels = batch_items
            # Transfer batch items to compute_device
            inp_chars, chars_len, mel, mel_len  = inp_chars.to(device), chars_len.to(device), mel.to(device), mel_len.to(device)
            speaker_id, stop_labels = speaker_id.to(device), stop_labels.to(device)
            
            # Feed inputs to the model
            postnet_outputs, outputs, stop_values, attn_weights = self.model(inp_chars, 
                                                                             chars_len, 
                                                                             mel, 
                                                                             mel_len, 
                                                                             speaker_id)
            # Compute loss
            loss, l1_loss, mse_loss, bce_loss = self._compute_loss(outputs, 
                                                                   postnet_outputs, 
                                                                   mel, 
                                                                   mel_len, 
                                                                   stop_values, 
                                                                   stop_labels)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            
            # Compute loss average and step duration for logger
            running_loss += loss.item()
            avg_loss = running_loss / itr
            step = self.model.get_step()

            # Write logs to the Tensorboard
            if step % 10 == 0:
                self.writer.add_scalar("train/l1_loss", l1_loss.item(), step)
                self.writer.add_scalar("train/mse_loss", mse_loss.item(), step)
                self.writer.add_scalar("train/bce_loss", bce_loss.item(), step)
                self.writer.add_scalar("train/loss", loss.item(), step)
            
            # Save checkpoints
            if step % self.config["chekpoint_save_steps"] == 0:
                self._save_checkpoint()

            # Save mel-spec and attention plots for the generated example
            if self.example_id in item_id:
                idx = item_id.index(self.example_id)
                plot_attention(attn_weights[idx][:, :].detach().cpu().numpy(), 
                               os.path.join(self.path_manager.example_path, f'{step}_attn'))
                example_mel = postnet_outputs.transpose(1, 2)[idx].detach().cpu().numpy()
                plot_spectrogram(example_mel, 
                                 os.path.join(self.path_manager.example_path, f'{step}_mel'))
            
            step_k = step // 1000
            msg = f'| Epoch: {epoch}: ({itr}/{total_iters}) | avg. loss: {avg_loss:#.4} | l1: {l1_loss:#.4} | ' + \
                  f'mse: {mse_loss:#.4} | bce: {bce_loss:#.4} | step: {step_k}k'
            stream(msg)

        # Save checkpoint after each epoch    
        self._save_checkpoint()

    def _eval_epoch(self, epoch):
        self.model.eval()
        device = self.device
        print(f"\nEvaluating epoch {epoch}:")
        with torch.no_grad():
            running_loss = 0
            total_iters = len(self.train_loader)
            # Train for one epoch
            for itr, (batch_items) in enumerate(self.eval_loader, 1):
                item_id, inp_chars, chars_len, mel, mel_len, speaker_id, stop_labels = batch_items
                # Transfer batch items to compute_device
                inp_chars, chars_len, mel, mel_len  = inp_chars.to(device), chars_len.to(device), mel.to(device), mel_len.to(device)
                speaker_id, stop_labels = speaker_id.to(device), stop_labels.to(device)
                
                # Feed inputs to the model
                postnet_outputs, outputs, stop_values, attn_weights = self.model(inp_chars, 
                                                                                chars_len, 
                                                                                mel, 
                                                                                mel_len, 
                                                                                speaker_id)
                # Compute loss
                loss, l1_loss, mse_loss, bce_loss = self._compute_loss(outputs, 
                                                                    postnet_outputs, 
                                                                    mel, 
                                                                    mel_len, 
                                                                    stop_values, 
                                                                    stop_labels)
                
                # Compute loss average and step duration for logger
                running_loss += loss.item()
                avg_loss = running_loss / itr

            print(f"Loss: {avg_loss}\n")
            self.writer.add_scalar("eval/loss", l1_loss.item(), epoch)
        self.model.train()

    def _save_checkpoint(self):
        k = self.model.get_step() // 1000
        checkpoint_path = os.path.join(self.path_manager.checkpoints_path, f"checkpoint_{k}K.pt")
        torch.save(self.model.state_dict(), checkpoint_path)


def main(args):
    # Load config
    config_file_path = os.path.join(args.config_path, "config.yml")
    config = load_config(config_file_path)
    
    # Set number of characters
    config["model"]["num_chars"] = len(char_list)
    
    # Path manager
    output_path = os.path.join(config["output_path"], config["run_name"])
    path_manager = PathManager(output_path)

    # Model
    model = Tacotron2(**config["model"])

    # Trainer
    trainer = TacotronTrainer(config, path_manager, model)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
