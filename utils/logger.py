import logging
import os

class Logger:
    def __init__(self, save_dir, is_main=True, verbose=True, filename="training.log", overwrite=True):
        self.is_main = is_main
        self.verbose = verbose
        self.log_path = os.path.join(save_dir, filename)

        if self.is_main and overwrite:
            os.makedirs(save_dir, exist_ok=True)

            logging.basicConfig(
                filename=self.log_path,
                filemode="w",
                level=logging.INFO,
                format="%(message)s",
            )
            self.logger = logging.getLogger()
        else:
            self.logger = None

    def log(self, msg, override=False):
        if self.is_main or override:
            if self.verbose or override:
                print(msg)
            if self.logger is not None:
                self.logger.info(msg)
