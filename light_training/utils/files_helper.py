
import os 
import glob 
import torch 

def delete_last_model(model_dir, symbol):

    last_model = glob.glob(f"{model_dir}/{symbol}*.pth")
    if len(last_model) != 0:
        os.remove(last_model[0])


def save_new_model_and_delete_last(model, optimizer, dice_score, epoch, save_path, scheduler = None, delete_symbol=None):
    save_dir = os.path.dirname(save_path)

    os.makedirs(save_dir, exist_ok=True)
    if delete_last_model is not None:
        delete_last_model(save_dir, delete_symbol)
    if scheduler is not None:
        save_state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'lr_scheduler': scheduler.state_dict(),
                    'dice_score': dice_score}
    else:
        save_state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'dice_score': dice_score}
    torch.save(save_state, save_path)

    print(f"model is saved in {save_path}")
