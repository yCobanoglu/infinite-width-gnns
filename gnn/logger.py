import json

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def log_training(self, epoch, loss_train, acc_train, loss_val=None, acc_val=None):
        self.writer.add_scalar("Loss/train", loss_train, epoch)
        self.writer.add_scalar("Accuracy/train", acc_train, epoch)
        loss_val is not None and self.writer.add_scalar("Loss/val", loss_val, epoch)
        acc_val is not None and self.writer.add_scalar("Accuracy/val", acc_val, epoch)

    def log_test(self, loss_test, acc_test):
        self.writer.add_scalar("Loss/test", loss_test)
        self.writer.add_scalar("Accuracy/test", acc_test)

    def log_dict(self, hp):
        json_hp = json.dumps(hp, indent=2)
        self.writer.add_text("hp", "".join("\t" + line for line in json_hp.splitlines(True)))

    def close(self):
        self.writer.close()
