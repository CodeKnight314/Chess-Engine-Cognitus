import argparse
from torch.utils.data import DataLoader, Dataset
import pickle
from model import Renatus
import torch.optim as optim 
import torch 
import os
from tqdm import tqdm
import numpy as np

class PGNDataset(Dataset):
    def __init__(self, txt_dir: str):
        super().__init__()
        self.txt_dir = txt_dir
        self.files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
        self.files.sort()  # Ensure consistent ordering
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        # Get file path for the current game state file
        state_file = os.path.join(self.txt_dir, self.files[index])
        
        # Read current and next states from file
        with open(state_file, "r") as f:
            lines = f.readlines()
            current_state_start = lines.index("# Current State\n") + 1
            next_state_start = lines.index("# Next State\n") + 1
            current_state = np.loadtxt(lines[current_state_start:next_state_start - 1]).reshape(12, 8, 8)
            next_state = np.loadtxt(lines[next_state_start:]).reshape(12, 8, 8)
        
        # Convert to torch tensors
        current_state_tensor = torch.from_numpy(current_state).float()
        next_state_tensor = torch.from_numpy(next_state).float()
        
        return current_state_tensor, next_state_tensor
    
def supervised_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = os.path.join(args.path, "saved_weights/")
    os.makedirs(output_path, exist_ok=True)
    
    model = Renatus(12, 5).to(device)  # Ensure model is moved to the appropriate device
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    
    criterion = torch.nn.MSELoss()
    
    train_dataset = PGNDataset(args.pkl)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count()//2)
    
    for epoch in range(args.epochs):
        model.train() 
        
        total_loss = 0.0
        for i, data in enumerate(tqdm(train_dl, desc=f"[Training Renatus][{epoch+1}/{args.epochs}]")):
            opt.zero_grad()
            state, next_state = data
            state, next_state = state.to(device), next_state.to(device)
            
            prediction = model(state)
            
            loss = criterion(prediction, next_state)
            loss.backward() 
            opt.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dl)
        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.6f}")
        
        # Save the model periodically
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_path, f"Renatus_epoch{epoch+1}.pth"))
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_path, "Renatus_final.pth"))
    print("Training completed and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Renatus model on chess PGN data.")
    parser.add_argument('--pkl', type=str, required=True, help='Path to the pickled training data')
    parser.add_argument('--path', type=str, default='./', help='Directory to save model weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval (in epochs) to save model weights')
    
    args = parser.parse_args()
    
    supervised_train(args)