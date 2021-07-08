import numpy as np
import torch
from torch.utils import data

class LineDataset(data.Dataset):
    """Line Fitting dataset.
    """

    def __init__(self, mode="train", numpoints=100, numdim=2, ratio_outlier=0.4):
        """Initialization.
        """
        self.mode = mode
        self.numdim = numdim
        self.ratio_outlier = ratio_outlier
        self.numpoints = numpoints

    def __len__(self):
        """ Returns number of samples. """

        # Just say we have 1M datapoints
        return int(1e6)

    def __getitem__(self, index):
        """Function to grab one data sample 

        Parameters
        ----------

        index: int
            Index to the sample that we are trying to extract.


        Returns
        -------

        x: torch.Tensor
            Point cloud with size of Bx2xN 

        y: torch.Tensor
            Line parameters with size of Bx3
        """

        # We'll only consider the 2D case for ease in many things
        assert self.numdim == 2

        with torch.no_grad():

            # Create random points
            pts = torch.rand(
                2, self.numdim,
                dtype=torch.float32
            ) * 2.0 - 1.0
            ones = torch.ones(2, 1, dtype=torch.float32)
            # Compute the line equation
            pts = torch.cat([pts, ones], dim=1)
            XXt = torch.matmul(pts.t(), pts)
            U, S, V = torch.svd(XXt)
            y = U[:, -1]

            # Create outliers
            outliers = torch.rand(
                self.numpoints, self.numdim,
                dtype=torch.float32
            ) * 2.0 - 1.0

            # Create inliers
            inliers = torch.rand(
                self.numpoints, self.numdim,
                dtype=torch.float32
            ) * 2.0 - 1.0

            # Project points onto the plane
            xx = torch.cat([
                inliers,
                torch.ones(self.numpoints, 1, dtype=torch.float32)
            ], dim=1)
            d = torch.matmul(xx, y) / torch.sum(y[:-1]**2)
            perp = d[..., None] * y[None, :-1]
            inliers = inliers - perp

            # Create selection mask
            mask = torch.rand(
                self.numpoints) >= self.ratio_outlier
            mask = mask[:, None].float()

            # Select inliers and outliers
            x = (1.0 - mask) * outliers + mask * inliers

            # Make D1K
            x = x.t()

        return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_data = LineDataset()

    # Create data loader for training and validation.
    tr_data_loader = data.DataLoader(dataset=train_data, batch_size=32)
    for x, y in tr_data_loader:
        """
        x: pts; y: line parameters 
        """
        print(f"x shape: {x.shape}; y shape: {y.shape}") 
        # Calculate the pts-level label
        X = torch.cat(
            [x, torch.ones(x.shape[0], 1, x.shape[2]).to(x.device)],
            dim=1)

        # err: BN1 
        err = torch.abs(
            torch.matmul(
                torch.transpose(X, 1, 2),
                y[..., None])) / torch.sum(y[..., None]**2, dim=1, keepdim=True)
        th_err = 0.1 
        labels = (err < th_err).float()
        
        # visualize an example
        idx = 0
        pts = x[idx].transpose(1, 0).numpy() 
        xs = pts[:, 0]
        ys = pts[:, 1]
        label = labels[idx].squeeze().numpy()
        plt.scatter(xs, ys, c=label, s=120, cmap="jet", marker=".", alpha=0.9, edgecolor="black")
        plt.savefig("vis.png", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
        break