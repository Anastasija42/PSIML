import torch
from tqdm import tqdm
import torch.optim as optim
from model import UNET
from UNetWithResnet50Encoder import UNetWithResnet50Encoder
from torchvision import transforms
import matplotlib.pyplot as plt
from dataloaders import train_dataloader, valid_dataloader, test_dataloader
from loss import depth_criterion, grad_criterion, normal_criterion, imgrad
from parameters import DEVICE, ONLY_MSE_LOSS, LEARNING_RATE, NUM_EPOCHS, MODEL_PATH, TRAIN_MODEL, LOAD_MODEL, GRADIENT_FACTOR, NORMAL_FACTOR


train_loss = []
val_loss = []
test_loss = []


def train_fn(loader, model, optimizer, loss_fn):
    for batch in tqdm(loader):
        images = batch[0]
        depths = batch[1]
        images = images.to(device=DEVICE, dtype=torch.float32)
        depths = depths.to(device=DEVICE, dtype=torch.float32)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, depths)
        if not ONLY_MSE_LOSS:
            grad_real, grad_fake = imgrad(depths), imgrad(predictions)
            loss += grad_criterion(grad_fake, grad_real) * GRADIENT_FACTOR
            loss += normal_criterion(grad_fake, grad_real) * NORMAL_FACTOR

        loss.backward()
        optimizer.step()

        train_loss.append(float(loss.detach().item()))


def val_fn(loader, model, loss_fn, epoch_idx):
    for idx, batch in enumerate(tqdm(loader)):
        images = batch[0]
        depths = batch[1]
        images = images.to(device=DEVICE, dtype=torch.float32)
        depths = depths.to(device=DEVICE, dtype=torch.float32)

        predictions = model(images)
        loss = loss_fn(predictions, depths)
        val_loss.append(float(loss.detach().item()))

        transform = transforms.ToPILImage()
        pred = transform(predictions[0].detach())
        img = transform(images[0].detach())
        dpt = transform(depths[0].detach())

        if idx % 1000 == 0:
            img.save(f"validation{epoch_idx}/image{idx}.jpeg", "JPEG")
            pred.save(f"validation{epoch_idx}/depth_estimation{idx}.png", "PNG")
            dpt.save(f"validation{epoch_idx}/real_depth{idx}.jpeg", "JPEG")


def test_fn(loader, model, loss_fn):
    for idx, batch in enumerate(tqdm(loader)):
        images = batch[0]
        depths = batch[1]
        images = images.to(device=DEVICE, dtype=torch.float32)
        depths = depths.to(device=DEVICE, dtype=torch.float32)

        predictions = model(images)
        loss = loss_fn(predictions, depths)
        test_loss.append(float(loss.detach().item()))

        transform = transforms.ToPILImage()
        pred = transform(predictions[0].detach())
        img = transform(images[0].detach())
        dpt = transform(depths[0].detach())

        if idx % 1000 == 0:
            img.save(f"test/image{idx}.jpeg", "JPEG")
            pred.save(f"test/depth_estimation{idx}.png", "PNG")
            dpt.save(f"test/real_depth{idx}.jpeg", "JPEG")


def main():
    model = UNetWithResnet50Encoder()
    model.to(device=DEVICE)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for idx in range(NUM_EPOCHS):
        if TRAIN_MODEL:
            print(f"Epoch number {idx}: ")
            model.train()
            train_fn(train_dataloader, model, optimizer, depth_criterion)

        model.eval()
        val_fn(valid_dataloader, model, depth_criterion, idx)

    model.eval()
    test_fn(test_dataloader, model, depth_criterion)

    plt.plot(train_loss)
    plt.show()

    plt.plot(val_loss)
    plt.show()

    plt.plot(test_loss)
    plt.show()

    if TRAIN_MODEL:
        torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
