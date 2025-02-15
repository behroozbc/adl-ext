import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image
from datasets import load_dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--idm',type=str)
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, store_path, image_size=32):
    """
    Generate and save images using the trained diffusion model.
    """
    model.eval()
    with torch.no_grad():
        images = diffusor.sample(model, image_size=image_size, batch_size=n_images, channels=3)
        images = (images.clamp(-1, 1) + 1) / 2  # Scale back to [0, 1]
        save_image(images, os.path.join(store_path, "generated_images.png"))
    print(f"Generated images saved to {store_path}")

def test(model, testloader, diffusor, device, args, save_path):
    """
    Test the model on a test dataset.
    
    1. Compute loss over the test set at specific timesteps.
    2. Generate and save images.
    """
    model.eval()
    total_loss = 0.0
    timesteps = args.timesteps
    save_images = True  # Generate visualizations

    pbar = tqdm(testloader, desc="Testing")
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")
        total_loss += loss.item()

        if save_images and step == 0:  # Save generated images for the first batch
            sample_and_save_images(8, diffusor, model, device, save_path)
            save_images = False  # Only generate once

    avg_loss = total_loss / len(testloader)
    print(f"Test set average loss: {avg_loss:.6f}")
def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


# def test(args):
#     # TODO (2.2): implement testing functionality, including generation of stored images.
#     pass


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('datasets', download=True, train=True, transform=transform)
    if args.idm is not None:
        dataset=load_dataset("huggan/smithsonian_butterflies_subset", cache_dir=f"/proj/ciptmp/{args.idm}/datasets/", split="train")
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('datasets', download=True, train=False, transform=transform)
    if args.idm is not None:
        dataset=load_dataset("huggan/smithsonian_butterflies_subset", cache_dir=f"/proj/ciptmp/{args.idm}/datasets/", split="test")
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = "/ext2/output"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path)
    torch.save(model.state_dict(), os.path.join("/ext2/models", args.run_name, f"ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
