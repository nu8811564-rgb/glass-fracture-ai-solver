import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. AI Model Architecture ---
class GlassAI(nn.Module):
    def __init__(self):
        super(GlassAI, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layer(x)

# --- 2. Training a quick model for demo ---
def prepare_model():
    # Synthetic data for training
    h = np.random.uniform(1, 20, 5000)
    a = np.random.uniform(0, 90, 5000)
    d = (np.sqrt(19.62 * h) * np.sin(np.radians(a))) + np.random.normal(0, 0.2, 5000)
    
    X = np.stack([h, a], axis=1)
    y = d.reshape(-1, 1)
    
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    
    model = GlassAI()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_t = torch.FloatTensor(scaler_X.transform(X))
    y_t = torch.FloatTensor(scaler_y.transform(y))
    
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = prepare_model()

# --- 3. Solver Function ---
def solve(target_dist):
    target_scaled = scaler_y.transform([[target_dist]])
    best_loss = float('inf')
    best_params = (0, 0)
    
    for _ in range(500):
        h_t, a_t = np.random.uniform(1, 20), np.random.uniform(0, 90)
        inp = torch.FloatTensor(scaler_X.transform([[h_t, a_t]]))
        pred = model(inp).item()
        if abs(pred - target_scaled[0][0]) < best_loss:
            best_loss = abs(pred - target_scaled[0][0])
            best_params = (h_t, a_t)
            
    h, a = best_params
    
    # Create a simple plot
    fig = plt.figure(figsize=(5, 4))
    plt.scatter(np.random.normal(target_dist, 0.5, 20), np.zeros(20), c='blue', alpha=0.5)
    plt.title(f"Impact Zone at {target_dist}m")
    plt.xlim(0, 20)
    plt.xlabel("Distance (meters)")
    
    result_text = f"Drop Height: {h:.2f} meters\nDrop Angle: {a:.2f} degrees"
    return result_text, fig

# --- 4. Gradio Interface ---
demo = gr.Interface(
    fn=solve,
    inputs=gr.Number(label="Target Distance (meters)", value=2.0),
    outputs=[gr.Textbox(label="AI Recommendation"), gr.Plot(label="Impact Visualization")],
    title="Glass Fracture AI Solver",
    description="Enter the distance where you want the glass to break. The AI will calculate the drop height and angle."
)

demo.launch()