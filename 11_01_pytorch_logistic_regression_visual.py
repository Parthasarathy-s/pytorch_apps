import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Tiny dataset: 6 points, 2 features, 2 classes
X = torch.tensor([[1.0, 1.0], [1.5, 2.0], [2.0, 1.0],
                  [4.0, 4.0], [4.5, 3.5], [5.0, 5.0]])
y = torch.tensor([0., 0., 0., 1., 1., 1.])

model = nn.Linear(2, 1)                 # z = w1*x1 + w2*x2 + b
criterion = nn.BCEWithLogitsLoss()      # log-loss from logits
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def show_params(tag):
    w = model.weight.data.flatten().tolist()
    b = model.bias.data.item()
    print(f"    {tag:<22} w=[{w[0]:+.4f}, {w[1]:+.4f}]  b={b:+.4f}")

losses, snapshots = [], {}
EPOCHS = 30
PRINT_DETAIL = {0, 1, 2, 29}

for epoch in range(EPOCHS):
    detail = epoch in PRINT_DETAIL
    if detail:
        print(f"\n{'=' * 60}\nEPOCH {epoch}\n{'=' * 60}")
        show_params("weights BEFORE")

    z = model(X).squeeze()                          # 1) forward
    if detail:
        print("\n[1] z = model(X)   (raw scores / logits)")
        print("    z    =", [round(v, 3) for v in z.tolist()])
        print("    p    =", [round(v, 3) for v in torch.sigmoid(z).tolist()],
              " <- sigmoid(z), only for our eyes")
        print("    y    =", y.tolist())

    loss = criterion(z, y)                          # 2) loss
    if detail:
        print(f"\n[2] loss = criterion(z, y)  ->  {loss.item():.4f}")

    optimizer.zero_grad()                           # 3) clear old grads
    if detail:
        print("\n[3] optimizer.zero_grad()  (grads reset; otherwise they accumulate)")

    loss.backward()                                 # 4) gradients
    if detail:
        gw = model.weight.grad.flatten().tolist()
        gb = model.bias.grad.item()
        print(f"\n[4] loss.backward()")
        print(f"    dLoss/dw = [{gw[0]:+.4f}, {gw[1]:+.4f}]   dLoss/db = {gb:+.4f}")

    optimizer.step()                                # 5) update
    if detail:
        print("\n[5] optimizer.step()   new = old - lr * grad   (lr=0.1)")
        show_params("weights AFTER")

    losses.append(loss.item())
    if epoch in (0, 5, 29):
        snapshots[epoch] = (model.weight.data.clone().flatten(), model.bias.data.item())

print("\nLoss over all epochs:")
print("  ", [round(l, 3) for l in losses])

# ---- Plots ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(losses, marker="o", ms=3)
axes[0].set_title("Loss per epoch")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("log-loss")

axes[1].scatter(X[:3, 0], X[:3, 1], c="tab:blue", label="class 0", s=70)
axes[1].scatter(X[3:, 0], X[3:, 1], c="tab:red", label="class 1", s=70)
xs = torch.linspace(0, 6, 50)
for ep, (w, b) in snapshots.items():
    ys = -(w[0] * xs + b) / w[1]            # where z = 0
    axes[1].plot(xs, ys, label=f"boundary @ epoch {ep}")
axes[1].set_ylim(0, 6)
axes[1].set_title("Decision boundary (z = 0) moving")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("training_trace.png", dpi=130)
