import torch
from tqdm import tqdm
import pickle

def train(model, train_loader, val_loader, optimizer, device, epochs=5, checkpoint_dir="checkpoints"):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch+1}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f"{checkpoint_dir}/text2gloss_epoch{epoch+1}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, f)
        print(f"✅ Saved checkpoint: {checkpoint_path}")
