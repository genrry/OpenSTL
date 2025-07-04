import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        
        batch_x_np = batch_x.cpu().numpy()
        pred_y_np = pred_y.cpu().numpy()
        batch_y_np = batch_y.cpu().numpy()
        
        # Compute metrics for this batch immediately to reduce memory usage
        batch_eval_res, _ = metric(pred_y_np, batch_y_np,
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))
        
        # Store only essential information for final aggregation
        outputs = {
            'batch_size': batch_x.shape[0],
            'mae': batch_eval_res['mae'] * batch_x.shape[0], 
            'mse': batch_eval_res['mse'] * batch_x.shape[0],
        }
        
        # Only store a small sample of the data for visualization
        if batch_idx == 0 and self.trainer.is_global_zero:
            outputs.update({
                'sample_inputs': batch_x_np[:4],
                'sample_preds': pred_y_np[:4],
                'sample_trues': batch_y_np[:4]
            })
        
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        # Aggregate metrics across all batches
        total_samples = sum([batch['batch_size'] for batch in self.test_outputs])
        total_mae = sum([batch['mae'] for batch in self.test_outputs]) / total_samples
        total_mse = sum([batch['mse'] for batch in self.test_outputs]) / total_samples
        
        eval_res = {'mae': total_mae, 'mse': total_mse}
        eval_log = f"mae:{eval_res['mae']:.4f}, mse:{eval_res['mse']:.4f}"
        
        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))
            
            metrics_data = np.array([eval_res['mae'], eval_res['mse']])
            np.save(osp.join(folder_path, 'metrics.npy'), metrics_data)
            
            sample_batch = next((batch for batch in self.test_outputs if 'sample_inputs' in batch), None)
            if sample_batch:
                for data_type in ['sample_inputs', 'sample_preds', 'sample_trues']:
                    np.save(osp.join(folder_path, data_type.replace('sample_', '') + '_sample.npy'), 
                           sample_batch[data_type])
                    
        self.test_outputs.clear()
        
        return {'mae': eval_res['mae'], 'mse': eval_res['mse']}