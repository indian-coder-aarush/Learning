import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==== Data ====
N = 500
X = np.linspace(-np.pi, np.pi, N).reshape(-1, 1).astype(np.float32)
y = np.sin(X).astype(np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)


# ==== Helper: train and return predictions ====
def train_and_predict(activation, title, epochs=3000, hidden=8, lr=0.01):
    # Small network
    model = nn.Sequential(
        nn.Linear(1, hidden),
        activation,
        nn.Linear(hidden,hidden),
        activation,
        nn.Linear(hidden, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"[{title}] Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

    # Predict
    model.eval()
    with torch.no_grad():
        y_hat = model(X_tensor).numpy()

    # Plot individual result
    plt.figure(figsize=(8, 5))
    plt.plot(X, y, label="True sin(x)", color="black", linewidth=2)
    plt.plot(X, y_hat, label=f"{title} Prediction", color="red")
    plt.title(f"Sine Approximation using {title}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_hat


# ==== Run individually ====
y_hat_relu = train_and_predict(nn.ReLU(), "ReLU",hidden = 3)
y_hat_sig = train_and_predict(nn.Sigmoid(), "Sigmoid", hidden = 3)
y_hat_tanh = train_and_predict(nn.Tanh(), "Tanh",hidden = 3)
