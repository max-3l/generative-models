import time
import torch
import torch.nn as nn
import pytest

class PairwiseMSELoss(nn.Module):
    """
    Computes loss: for each i in batch,
        d_i = sum_n mse(x_i, x_n) * mse(y_i, y_n),
    then returns -mean(d_i).
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction='none') -> torch.Tensor:
        B = x.size(0)
        # flatten to (B, F)
        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)

        # pairwise differences
        x_i = x_flat.unsqueeze(1)  # (B, 1, F)
        x_n = x_flat.unsqueeze(0)  # (1, B, F)
        y_i = y_flat.unsqueeze(1)
        y_n = y_flat.unsqueeze(0)

        diff_x = (x_i - x_n).pow(2)  # (B, B, F)
        diff_y = (y_i - y_n).pow(2)

        # mean over feature dims => (B, B)
        mse_x = diff_x.mean(dim=2)
        mse_y = diff_y.mean(dim=2)

        # compute d_i and average
        d = (mse_x * mse_y).sum(dim=1)
        if reduction == 'mean':
            return -d.mean()
        elif reduction == 'sum':
            return -d.sum()
        else:
            # No reduction
            return -d

class PairwiseL1Loss(nn.Module):
    """
    Computes loss: for each i in batch,
        d_i = sum_n mse(x_i, x_n) * mse(y_i, y_n),
    then returns -mean(d_i).
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction='none') -> torch.Tensor:
        B = x.size(0)
        # flatten to (B, F)
        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)

        # pairwise differences
        x_i = x_flat.unsqueeze(1)  # (B, 1, F)
        x_n = x_flat.unsqueeze(0)  # (1, B, F)
        y_i = y_flat.unsqueeze(1)
        y_n = y_flat.unsqueeze(0)

        diff_x = (x_i - x_n).abs()  # (B, B, F)
        diff_y = (y_i - y_n).abs()

        # mean over feature dims => (B, B)
        mse_x = diff_x.mean(dim=2)
        mse_y = diff_y.mean(dim=2)

        # compute d_i and average
        d = (mse_x * mse_y).sum(dim=1)

        if reduction == 'mean':
            return -d.mean()
        elif reduction == 'sum':
            return -d.sum()
        else:
            # No reduction
            return -d

def pairwise_l1_product_loss(x_batch, y_batch, reduction='none'):
    """
    Computes L = -mean(sum_n(l1(x_i, x_n) * l1(y_i, y_n))).

    Args:
        x_batch (Tensor): Predictions, shape (N, D).
        y_batch (Tensor): Targets, shape (N, D).

    Returns:
        Tensor: Scalar loss value.
    """
    N = x_batch.shape[0]
    # Handle edge case of batch size 1 or 0 to avoid errors
    if N <= 1:
        # Return zero loss or raise an error, as pairwise distances are trivial
        return torch.tensor(0.0, device=x_batch.device, dtype=x_batch.dtype, requires_grad=True)
    x_batch = x_batch.view(N, -1)  # Flatten to (N, D)
    y_batch = y_batch.view(N, -1)  # Flatten to (N, D)
    # Compute pairwise L1 distance matrices (N, N)
    # torch.cdist is efficient for computing all pairs between two sets,
    # using it with the same input computes the required N x N matrix.
    dist_x = torch.cdist(x_batch, x_batch, p=1) #
    dist_y = torch.cdist(y_batch, y_batch, p=1) #

    # Compute element-wise product of distance matrices
    dist_prod = dist_x * dist_y

    # Sum products over n for each i
    # Sum along dimension 1 (columns) to get the sum for each row i
    d = torch.sum(dist_prod, dim=1) # Shape (N,)

    if reduction == 'mean':
        return -d.mean()
    elif reduction == 'sum':
        return -d.sum()
    else:
        # No reduction
        return -d

def naive_pairwise_l1_loss(x: torch.Tensor, y: torch.Tensor, reduction='none') -> torch.Tensor:
    """
    Naive loop-based implementation (no vectorization).
    """
    B = x.size(0)
    # flatten features
    x_flat = x.view(B, -1)
    y_flat = y.view(B, -1)
    d_vals = []
    for i in range(B):
        d_i = 0.0
        for n in range(B):
            # compute MSE between x_i and x_n
            diff_x = (x_flat[i] - x_flat[n]).abs()
            mse_x = diff_x.mean()
            diff_y = (y_flat[i] - y_flat[n]).abs()
            mse_y = diff_y.mean()
            d_i += mse_x * mse_y
        d_vals.append(d_i)
    d_tensor = torch.stack(d_vals)
    if reduction == 'mean':
        return -d_tensor.mean()
    elif reduction == 'sum':
        return -d_tensor.sum()
    else:
        # No reduction
        return -d_tensor

class PairwiseL1LossCdist(nn.Module):
    """
    L1 implementation using torch.cdist:
    Uses pairwise L1 distances to compute MAE.
    """
    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction='none') -> torch.Tensor:
        B, F = x.size(0), x.view(x.size(0), -1).size(1)
        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)

        dist_x = torch.cdist(x_flat, x_flat, p=1)  # (B, B)
        dist_y = torch.cdist(y_flat, y_flat, p=1)
        mae_x = dist_x / F
        mae_y = dist_y / F

        d = (mae_x * mae_y).sum(dim=1)
        if reduction == 'mean':
            return -d.mean()
        elif reduction == 'sum':
            return -d.sum()
        else:
            # No reduction
            return -d

# ----------- Test cases ----------- #

def test_pairwise_loss_basic_scalar():
    # Simple 2-item batch with 1-dim features
    x = torch.tensor([[0.0], [2.0]], requires_grad=True)
    y = torch.tensor([[0.0], [2.0]])
    loss_fn = PairwiseMSELoss()
    loss = loss_fn(x, y, reduction='mean')
    # Manually: for i=0: d0 = 0*0 + (4)*(4) = 16
    #          for i=1: same => d1 = 16
    # mean(d) = 16, so loss = -16
    assert torch.allclose(loss, torch.tensor(-16.0))

    # check gradient w.r.t x[0] (only from cross term)
    loss.backward()
    # d0 contributed gradient: grad_x0 = d(d0)/dx0 = d(mse_x[0,1]*mse_y[0,1])/dx0
    # mse_x[0,1] = (0-2)^2 = 4, derivative = 2*(x0-2)/1 = -4
    # so grad loss = -(1/2)*(mse_y*derivative) = -(1/2)*(4*-4) = 8
    # But both i=0 and i=1 contribute: gradient sum = 8 + ?
    # Here we just check gradient is non-zero tensor
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_pairwise_loss_zero_batch():
    # If batch size = 1, loss should be zero (only self term zero)
    x = torch.randn(1, 5, requires_grad=True)
    y = torch.randn(1, 5)
    loss = PairwiseMSELoss()(x, y)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_pairwise_loss_symmetry():
    # Loss should be invariant to swapping batch items
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4)
    loss_fn = PairwiseMSELoss()
    loss1 = loss_fn(x, y)

    # swap indices 0 and 1
    idx = [1, 0, 2]
    loss2 = loss_fn(x[idx], y[idx], reduction='mean')
    assert torch.allclose(loss1, loss2)


def test_pairwise_loss_implementations():
    # Test that the two implementations give the same result
    x = torch.randn(5, 4, requires_grad=True)
    y = torch.randn(5, 4)

    loss_fn1 = PairwiseL1Loss()

    loss1 = loss_fn1(x, y)
    loss2 = pairwise_l1_product_loss(x, y)

    assert torch.allclose(loss1, loss2)

def test_pairwise_loss_implementations_naive():
    # Test that the two implementations give the same result
    x = torch.randn(5, 4, requires_grad=True)
    y = torch.randn(5, 4)

    loss_fn1 = PairwiseL1Loss()

    loss1 = loss_fn1(x, y)
    loss2 = naive_pairwise_l1_loss(x, y)

    assert torch.allclose(loss1, loss2)

def test_pairwise_loss_implementations_cdist():
    # Test that the two implementations give the same result
    x = torch.randn(5, 4, requires_grad=True)
    y = torch.randn(5, 4)

    loss_fn1 = PairwiseL1Loss()

    loss1 = loss_fn1(x, y)
    loss2 = PairwiseL1LossCdist()(x, y)

    assert torch.allclose(loss1, loss2)

@pytest.mark.parametrize("reduction", ['none', 'mean', 'sum'])
def test_pairwise_loss_implementations_reductions(reduction):
    x = torch.randn(5, 4, requires_grad=True)
    y = torch.randn(5, 4)

    loss1 = naive_pairwise_l1_loss(x, y, reduction=reduction)
    loss2 = PairwiseL1Loss()(x, y, reduction=reduction)
    loss3 = pairwise_l1_product_loss(x, y, reduction=reduction)
    loss4 = PairwiseL1LossCdist()(x, y, reduction=reduction)

    assert torch.allclose(loss1, loss2, atol=1e-5)
    assert torch.allclose(loss1, loss3, atol=1e-5)
    assert torch.allclose(loss1, loss4, atol=1e-5)

def benchmark(batch_size=32, feature_dim=128, device='cpu'):
    x = torch.randn(batch_size, feature_dim, device=device)
    y = torch.randn(batch_size, feature_dim, device=device)
    reps = 10

    reductions = ['mean', 'sum', 'none']
    functions = {
        'naive': naive_pairwise_l1_loss,
        'vectorized': lambda x, y, reduction: PairwiseL1Loss()(x, y, reduction=reduction),
        'cdist': lambda x, y, reduction: PairwiseL1LossCdist()(x, y, reduction=reduction)
    }

    print(f"Benchmark (batch={batch_size}, feat={feature_dim}, device={device}):")
    for reduction in reductions:
        print(f"  Reduction: {reduction}")
        timings = {}
        for name, fn in functions.items():
            _ = fn(x, y, reduction=reduction)  # warm-up
            if device == 'cuda': torch.cuda.synchronize()
            start = time.time()
            for _ in range(reps):
                _ = fn(x, y, reduction=reduction)
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = (time.time() - start) / reps
            timings[name] = elapsed
        for name, t in timings.items():
            print(f"    {name}: {t*1000:.2f} ms")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
    # Run benchmarks
    benchmark(batch_size=32, feature_dim=128)
    benchmark(batch_size=64, feature_dim=256)
    if torch.cuda.is_available():
        benchmark(batch_size=32, feature_dim=128, device='cuda')
        benchmark(batch_size=64, feature_dim=256, device='cuda')
