import torch
from argparse import ArgumentParser
from vptq.utils.hessian import load_hessian
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--load_hessian_dir', type=str, default=None,
                        help='Directory containing Hessian .pt files')
    parser.add_argument('--store_inv_hessian_dir', type=str, default=None,
                        help='Directory to save inverted Hessian .pt files')

    args = parser.parse_args()

    # create folder
    os.makedirs(args.store_inv_hessian_dir, exist_ok=True)

    percdamp = 0.01
    hessian_files = [f for f in os.listdir(
        args.load_hessian_dir) if f.endswith('.pt')]

    for hessian_file in hessian_files:
        hessian_path = os.path.join(args.load_hessian_dir, hessian_file)
        hessian, mu = load_hessian(hessian_path)
        dev = 'cuda'
        hessian = hessian.to(dev)

        zero_idx = torch.diag(hessian) == 0
        hessian[zero_idx, zero_idx] = 1

        # get permutation
        perm = torch.argsort(torch.diag(hessian), descending=True).to(dev)
        hessian = hessian[perm][:, perm]

        # add damping
        damp = percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(hessian.shape[0], device=dev)
        hessian[diag, diag] += damp

        # inverse Hessian
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        inv_hessian = hessian

        # Saving the inverted Hessian to the specified directory with the same file name
        save_path = os.path.join(args.store_inv_hessian_dir, hessian_file)
        torch.save({'invH': inv_hessian.to('cpu'),
                    'perm': perm.to('cpu'),
                    'zero_idx': zero_idx.to('cpu')}, save_path)
        print(f'Saved inverted Hessian to {save_path}')