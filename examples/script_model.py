import torch

class ScriptModel(torch.nn.Module):
    def __init__(self, 
                 model,
                 input_shape = (2, 3, 512, 512), 
                 mean = [0.405,0.432,0.397],
                 std = [0.164,0.173,0.153],
                 min = 0.0,
                 max = 1.0):
        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.mean = torch.tensor(mean).resize_(len(mean), 1)
        self.std = torch.tensor(std).resize_(len(std), 1)
        self.min_val = min
        self.max_val = max
        
        input_tensor = torch.rand(input_shape).to(self.device)
        self.model_scripted = torch.jit.trace(model, input_tensor)
    
    def forward(self, input):
        shape = input.shape
        B, C = shape[0], shape[1]
        x_min = 0
        x_max = 255
        input = (self.max_val - self.min_val) * (input - x_min) / (x_max - x_min) + self.min_val
        input = (input.view(B, C, -1) - self.mean) / self.std
        input = input.view(shape)
        return self.model_scripted(input)