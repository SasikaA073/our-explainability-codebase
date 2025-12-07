import os
import argparse

# Parse arguments FIRST
parser = argparse.ArgumentParser(description='Fine-tune DINOv2 on ImageNet')
parser.add_argument('--gpu', type=int, default=7, help='GPU index to use (default: 7)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation (default: 32)')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
args = parser.parse_args()

# --- CRITICAL: Set this BEFORE importing torch ---
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from datasets import load_dataset
import wandb
from tqdm import tqdm

# Keys
os.environ["WANDB_API_KEY"] = ""
os.environ["HF_TOKEN"] = ""

# 1. Initialize WandB
wandb.init(project="dinov2-finetune", name="dinov2-head-train")

os.makedirs("checkpoints", exist_ok=True)
best_model_path = "checkpoints/dinov2_finetuned_best.pth"
best_val_acc = 0.0

try:
    # 2. Model Setup
    model_id = "vit_base_patch14_reg4_dinov2.lvd142m"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = timm.create_model(model_id, pretrained=True)

    # Freeze Backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace Head
    NUM_CLASSES = 1000
    model.head = nn.Linear(
        in_features=model.blocks[-1].mlp.fc2.out_features, 
        out_features=NUM_CLASSES, 
        bias=True
    )
    model.to(device)

    # 3. Data & Transforms
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    transforms_train = timm.data.create_transform(**data_config, is_training=True)
    transforms_val = timm.data.create_transform(**data_config, is_training=False)

    # --- Transforms ---
    def apply_train_transforms(examples):
        examples['pixel_values'] = [transforms_train(image.convert("RGB")) for image in examples['image']]
        return examples

    def apply_val_transforms(examples):
        examples['pixel_values'] = [transforms_val(image.convert("RGB")) for image in examples['image']]
        return examples

    # --- Custom Collate Function ---
    def collate_fn(batch):
        pixel_values = torch.stack([torch.tensor(item['pixel_values']) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'pixel_values': pixel_values, 'label': labels}

    # Using Streaming
    print("Loading dataset stream...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    val_ds_stream = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    # Shuffle and Subset - FIXED: Using 10000 and 1000 samples
    train_dataset = dataset.shuffle(seed=42, buffer_size=10000).take(10000) 
    val_dataset = val_ds_stream.take(1000) 

    # Map Transforms
    train_dataset = train_dataset.map(apply_train_transforms, batched=True)
    val_dataset = val_dataset.map(apply_val_transforms, batched=True)

    # Remove original columns
    train_dataset = train_dataset.remove_columns(["image"])
    val_dataset = val_dataset.remove_columns(["image"])

    # 4. DataLoaders with collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # 5. Training Setup
    optimizer = optim.AdamW(model.head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 6. Training & Validation Loop
    num_epochs = int(args.epochs)
    print("Starting Training...")

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in pbar:
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            train_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            wandb.log({"train_loss_step": loss.item()})
            pbar.set_postfix({"loss": loss.item()})

        epoch_acc = 100 * correct / total
        epoch_loss = train_loss / train_batches
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

        print(f"Epoch {epoch+1}: Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_accuracy": epoch_acc,
            "train_loss_epoch": epoch_loss,
            "val_accuracy": val_acc,
            "val_loss": avg_val_loss,
            "best_val_accuracy": best_val_acc
        })

    print("Saving final model...")
    final_model_path = "dinov2_finetuned_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path} with Val Acc: {best_val_acc:.2f}%")

    # Upload both models to WandB
    best_artifact = wandb.Artifact('model-weights-best', type='model')
    best_artifact.add_file(best_model_path)
    wandb.log_artifact(best_artifact)
    
    final_artifact = wandb.Artifact('model-weights-final', type='model')
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)

except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Finishing WandB run...")
    wandb.finish()
