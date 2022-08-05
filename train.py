import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from torchvision import transforms
import numpy as np
from NYU_dataset import MyDataset2, MyDataset
import matplotlib.pyplot as plt
from uh import train_infos, val_infos

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 5
NUM_WORKERS = 0
IMAGE_HEIGHT = 480 
IMAGE_WIDTH = 640  
IMAGES_PATH = 'C:/Users/psiml8/Documents/GitHub/PSIML/npy/images.npy'
DEPTHS_PATH = 'C:/Users/psiml8/Documents/GitHub/PSIML/npy/depths.npy'
MODEL_PATH = 'save_model.pt'
LOAD_MODEL = False

total_train_loss = []
total_val_loss = []

def train_fn(loader, model, optimizer, loss_fn):
    total_loss = 0

    for batch in tqdm(loader):
        images = batch[0]
        depths = batch[1]
        images = images.to(device = DEVICE, dtype = torch.float32)
        depths = depths.to(device = DEVICE, dtype = torch.float32)
            
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, depths)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.detach().item())
        total_train_loss.append(float(loss.detach().item()))
        
    return total_loss  #kako?

def val_fn(loader, model, loss_fn):
    total_loss = 0
    for idx, batch in enumerate(tqdm(loader)):
        images = batch[0]
        depths = batch[1]
        images = images.to(device = DEVICE, dtype = torch.float32)
        depths = depths.to(device = DEVICE, dtype = torch.float32)
            
        predictions = model(images)
        loss = loss_fn(predictions, depths)
        total_loss += float(loss.detach().item())
        total_val_loss.append(float(loss.detach().item()))

        transform = transforms.ToPILImage()
        pred = transform(predictions[0].detach())
        img = transform(images[0].detach())
        dpt = transform(depths[0].detach())

        img.save(f"validation1/image{idx}.jpeg", "JPEG")
        pred.save(f"validation1/depth_estimation{idx}.png", "PNG")
        dpt.save(f"validation1/real_depth{idx}.jpeg", "JPEG")
        
    return total_loss 




def main():
    
    
    model = UNET()
    model.to(device=DEVICE)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    images = np.load(IMAGES_PATH)
    depths = np.load(DEPTHS_PATH)
    data = np.array(list(zip(images, depths)))

    data_test = data

    train_set=MyDataset2(train_infos, True)
    val_set=MyDataset2(val_infos, False)
    test_set=MyDataset(data_test)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    valid_dataloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_train_losses = []
    all_val_losses = []
    for idx in range(NUM_EPOCHS):
        if not LOAD_MODEL:
            print(f"Epoch number {idx}: ")
            model.train()
            train_loss = train_fn(train_dataloader, model, optimizer, loss_fn)
            all_train_losses.append(train_loss)

        model.eval()
        val_loss = val_fn(valid_dataloader, model, loss_fn)
        all_val_losses.append(val_loss)
    
    plt.plot(all_train_losses)
    plt.show()

    plt.plot(all_val_losses)
    plt.show()

    plt.plot(total_train_loss)
    plt.show()

    plt.plot(total_val_loss)
    plt.show()
    
    torch.save(model.state_dict(), MODEL_PATH)

if __name__=="__main__":
    main()
