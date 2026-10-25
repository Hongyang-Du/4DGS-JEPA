import torch

from gsjepa.geometry import (affine_spd_distance, axis_angle_to_matrix, compose_transform, cov_to_log_cholesky,
                             relative_transform, rotation_angle, se3_distance)


def _apply(R, d, x):
    return (R @ x.unsqueeze(-1)).squeeze(-1) + d


def test_rodrigues_is_rotation_with_correct_angle():
    w = torch.randn(64, 3)
    R = axis_angle_to_matrix(w)
    I = torch.eye(3).expand(64, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), I, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(64), atol=1e-5)
    theta = w.norm(dim=-1)
    expected = torch.where(theta > torch.pi, 2 * torch.pi - theta, theta)  # angle wraps at pi
    assert torch.allclose(rotation_angle(R), expected, atol=1e-4)


def test_compose_applies_rightmost_first():
    R1, R2 = axis_angle_to_matrix(torch.randn(7, 3) * 0.3), axis_angle_to_matrix(torch.randn(7, 3) * 0.3)
    d1, d2, x = torch.randn(7, 3), torch.randn(7, 3), torch.randn(7, 3)
    Rc, dc = compose_transform(R1, d1, R2, d2)
    assert torch.allclose(_apply(Rc, dc, x), _apply(R2, d2, _apply(R1, d1, x)), atol=1e-5)


def test_relative_transform_maps_pose0_to_pose1():
    R0, R1 = axis_angle_to_matrix(torch.randn(5, 3)), axis_angle_to_matrix(torch.randn(5, 3))
    d0, d1, x = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)
    Rr, dr = relative_transform(R0, d0, R1, d1)
    assert torch.allclose(_apply(Rr, dr, _apply(R0, d0, x)), _apply(R1, d1, x), atol=1e-5)
    assert torch.allclose(se3_distance(R0, d0, R0, d0), torch.zeros(5), atol=1e-5)


def test_spd_coordinates_and_distance():
    A = torch.randn(10, 3, 3)
    cov = A @ A.transpose(-1, -2) + 0.1 * torch.eye(3)
    assert cov_to_log_cholesky(cov).shape == (10, 6)
    assert torch.allclose(affine_spd_distance(cov, cov), torch.zeros(10), atol=1e-3)
    # affine invariance: d(W A W^T, W B W^T) = d(A, B)
    B = cov.flip(0)
    W = torch.randn(10, 3, 3) + 3 * torch.eye(3)
    d1 = affine_spd_distance(cov, B)
    d2 = affine_spd_distance(W @ cov @ W.transpose(-1, -2), W @ B @ W.transpose(-1, -2))
    assert torch.allclose(d1, d2, rtol=1e-2, atol=1e-3)
