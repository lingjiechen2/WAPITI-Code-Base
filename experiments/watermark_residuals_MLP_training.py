import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(
    filename='/remote-home/miintern1/watermark-learnability/logs/MLP.txt',  # Specify the log file name
    level=logging.INFO,          # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log message format
)

class LogitsDataset(Dataset):
    def __init__(self, input_logits, target_logits):
        self.input_logits = input_logits
        self.target_logits = target_logits

    def __len__(self):
        return len(self.input_logits)

    def __getitem__(self, idx):
        return self.input_logits[idx], self.target_logits[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_result_folder = "/remote-home/miintern1/watermark-learnability/data/c4/"
watermark_residuals = torch.load(final_result_folder + "watermark_residuals.pt", map_location= device)
vanilla_residuals = torch.load(final_result_folder + "vanilla_residuals.pt", map_location=device)

batch_size = 64
train_ratio = 0.9
analyze_layers = [list(vanilla_residuals.keys())[-1]]
print(analyze_layers)

watermark_configs = {
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2":{"type": "kgw", "k": 0, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_0", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 1.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
     "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2":{"type": "kgw", "k": 1, "gamma": 0.25, "delta": 2.0, "seeding_scheme": "simple_1", "kgw_device": "cpu"},
}


dataset_dict = dict()
train_dataloader_dict = dict()
eval_dataloader_dict = dict()
for layer in analyze_layers:
    dataset_dict[layer] = dict()
    train_dataloader_dict[layer] = dict()
    eval_dataloader_dict[layer] = dict()
    for watermark_name, _ in watermark_configs.items():
        dataset_dict[layer][watermark_name] = LogitsDataset(vanilla_residuals[layer], watermark_residuals[watermark_name][layer])
        train_dataset, test_dataset = torch.utils.data.random_split(dataset_dict[layer][watermark_name], [int(train_ratio*len(dataset_dict[layer][watermark_name])), len(dataset_dict[layer][watermark_name]) - int(train_ratio*len(dataset_dict[layer][watermark_name]))])
        train_dataloader_dict[layer][watermark_name] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader_dict[layer][watermark_name] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class TransformModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=500, output_dim=300):
        super(TransformModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

# Training function
def train_model(model, dataloader, num_epochs=10, learning_rate=0.001, watermark_name=None, layer=None):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create a SummaryWriter to log metrics
    num_layers = len(model.layers)
    hidden_dimension = model.layers[0].out_features
    hyperparameter_name = f"num_layers_{num_layers}_hidden_dim_{hidden_dimension}_lr_{learning_rate}_epochs_{num_epochs}_layer_{layer}_watermark_{watermark_name}"
    # logging.info(f"Training model with hyperparameters: {hyperparameter_name}")
    writer = SummaryWriter(log_dir=f'runs/{hyperparameter_name}')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
     
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # progress_bar.set_postfix(loss=running_loss/(i+1))
        
        epoch_loss = running_loss / len(dataloader)
        # logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    # logging.info("Training complete.")
    writer.close()

def evaluate_model(model, dataloader):
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    
    num_layers = len(model.layers)
    hidden_dimension = model.layers[0].out_features
    hyperparameter_name = f"num_layers_{num_layers}_hidden_dim_{hidden_dimension}"
    writer = SummaryWriter(log_dir=f'runs/{hyperparameter_name}/evaluation')

    running_loss = 0.0
    # progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            # progress_bar.set_postfix(loss=running_loss/(i+1))
    
    eval_loss = running_loss / len(dataloader)

    writer.add_scalar('Evaluation Loss', eval_loss, 0)
    writer.close()
    return eval_loss

model_dimension = 4096
num_layers_list = [1,2,3]
hidden_dim_list = [model_dimension * i for i in [2,4]]
learning_rate_list = [0.01, 0.001, 0.0001]
epoch_list = [10, 20, 30]
print(f'{hidden_dim_list=}')


# # For testing
# model_dimension = 4096
# num_layers_list = [2]
# hidden_dim_list = [2]
# learning_rate_list = [0.01]
# epoch_list = [10]


print("Starting Training")
total_iterations = len(analyze_layers) * len(watermark_configs)
with tqdm(total=total_iterations, desc="Processing Layers and Watermarks") as pbar:
    for target_layer in analyze_layers:
        for target_watermark, _ in watermark_configs.items():
        # target_watermark =  "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
            logging.info(f"Training model for layer: {target_layer}, watermark: {target_watermark}")
            dataloader = train_dataloader_dict[target_layer][target_watermark]
            eval_dataloader = eval_dataloader_dict[target_layer][target_watermark]

            # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Hyperparameter search
            best_model = None
            best_loss = float('inf')
            best_hyperparameters = None
            for num_layers in num_layers_list:
                for hidden_dim in hidden_dim_list:
                    for learning_rate in learning_rate_list:
                        for num_epochs in epoch_list:
                            model = TransformModel(num_layers=num_layers, input_dim=model_dimension, hidden_dim=hidden_dim, output_dim=model_dimension)
                            train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, watermark_name=target_watermark, layer=target_layer)
                            # torch.save(model.state_dict(), f"model_{num_layers}_{hidden_dim}_{learning_rate}_{num_epochs}.pt")
                            eval_loss = evaluate_model(model, eval_dataloader)
                            if eval_loss < best_loss:
                                best_loss = eval_loss
                                best_model = model
                                best_hyperparameters = {
                                    "num_layers": num_layers,
                                    "hidden_dim": hidden_dim,
                                    "learning_rate": learning_rate,
                                    "num_epochs": num_epochs
                                }
                                # logging.info(f"Best model found with loss: {best_loss}, Parameter Set:{best_hyperparameters} ")
                    save_dict = {
                        "model_state_dict": best_model.state_dict(),
                        "hyperparameters": best_hyperparameters
                    }
                    torch.save(save_dict, f"/remote-home/miintern1/watermark-learnability/data/model_weights_2/{target_layer}_{target_watermark[40:]}_hidden_dim_{hidden_dim}_num_layers_{num_layers}_best_model.pt")
                    logging.info(f"Best model found with loss: {best_loss}, Parameter Set:{best_hyperparameters} ")     
            pbar.update(1)