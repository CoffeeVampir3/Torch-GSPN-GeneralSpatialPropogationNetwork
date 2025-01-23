import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from model.example_network import GSPNNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
import numpy as np
from utils.trainutils import count_parameters_layerwise

def get_dataloaders(batch_size=4096):
    dataset = load_dataset("cifar10")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
    ])
    
    def transform_dataset(examples):
        tensors = torch.stack([transform(img) for img in examples['img']])
        return {'pixel_values': tensors, 'label': examples['label']}
    
    train_dataset = dataset['train'].with_transform(transform_dataset)
    val_dataset = dataset['test'].with_transform(transform_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train(model, train_loader, val_loader, epochs=100):
    torch.set_float32_matmul_precision('high')
    
    device = torch.device('cuda')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = DistributedShampoo(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        epsilon=1e-12,
        weight_decay=1e-05,
        max_preconditioner_dim=2048,
        precondition_frequency=100,
        start_preconditioning_step=250,
        use_decoupled_weight_decay=False,
        grafting_config=AdamGraftingConfig(
            beta2=0.999,
            epsilon=1e-12,
        ),
    )
    
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=5e-6
    )
    
    print(count_parameters_layerwise(model))
    
    model = torch.compile(
        model,
        backend='inductor',
        dynamic=False,
        fullgraph=True,
    )
    
    # Very scuffed counting but it's an example network who cares
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Validation Accuracy: {acc:.2f}%')

if __name__ == "__main__":
    model = GSPNNetwork()
    train_loader, val_loader = get_dataloaders()
    train(model, train_loader, val_loader)