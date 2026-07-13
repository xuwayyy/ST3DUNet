from torch.utils.data import DataLoader
import time

class ImageDataModule:
    def __init__(
            self,
            train_dataset,
            val_dataset,
            test_dataset,
            global_batch_size,
            num_workers,
    ):
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.batch_size = global_batch_size
        print(f"Each GPU will receive {self.batch_size} images")

    def setup(self, stage=None):
        """
        Args:
            stage (str): stage of the datamodule
                Is be one of "fit" or "test" or None
                fit or None for set trainset and val set
                test for testset
        """

        print("Stage:", stage)
        start_time = time.time()

        if stage in ["fit", None]:
            self.train_dataset = self._builders["train"]
            self.val_dataset = self._builders["val"]
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")
        elif stage == "test":
            self.test_dataset = self._builders["test"]
            print(f"Test dataset size: {len(self.test_dataset)}")
        else:
            raise ValueError(f"Unknown stage: {stage}")

        print(f"Setup took {(time.time() - start_time):.2f} seconds")

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=self.test_dataset.collate_fn,
        )

    def num_classes(self):
        return self._builders["train"]().num_classes

