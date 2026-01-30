from bert.lr_scheduler import build_lr_scheduler_linear_warmup_then_cosine
import math
import torch


learning_rate = 1e-3
linear_start_factor = 0.1
linear_end_factor = 1.0
linear_steps = 10
cosine_min_lr = 1e-5
total_steps = 100

fake_optimizer = torch.optim.Adam(
    [torch.nn.Parameter(torch.zeros(1))], lr=learning_rate
)
scheduler = build_lr_scheduler_linear_warmup_then_cosine(
    fake_optimizer,
    linear_start_factor,
    linear_end_factor,
    linear_steps,
    cosine_min_lr,
    total_steps,
)

all_lr = list[float]()
for _ in range(100):
    lr = scheduler.get_last_lr()
    assert len(lr) == 1
    all_lr.append(lr[0])
    scheduler.step()

print(all_lr)

for i in range(linear_steps + 1):
    factor = (
        linear_end_factor - linear_start_factor
    ) * i / linear_steps + linear_start_factor
    assert abs(math.log(all_lr[i]) - math.log(factor * learning_rate)) < 1e-7

for i in range(linear_steps, total_steps):
    x = (i - linear_steps) / (total_steps - linear_steps)
    factor = math.cos(x * math.pi) / 2 + 0.5
    lr = factor * (learning_rate - cosine_min_lr) + cosine_min_lr
    assert abs(math.log(all_lr[i]) - math.log(lr)) < 1e-7


if False:
    import matplotlib.pyplot as plt
    plt.plot(range(len(all_lr)), all_lr)
    plt.savefig("./test.png")
