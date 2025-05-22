import torch
import time
import neural

device = torch.device("cuda")

dummy_input = torch.randn(32, state_dim).to(device)
model = neural.MarioNet(state_dim, action_dim).to(device)

start = time.time()
out = model(dummy_input)
torch.cuda.synchronize()  # wait for GPU
end = time.time()

print(f"Forward pass time: {end - start:.4f} seconds")