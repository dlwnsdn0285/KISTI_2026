"""AdamW 옵티마이저 (모델 가중치는 bf16, 옵티마이저 상태는 fp32).

옵티마이저가 가중치의 fp32 사본을 갖고 거기서 갱신한 뒤 bf16 가중치에 옮겨 넣는다.
학습 후반에 학습률이 작아져도 작은 갱신이 bf16 반올림으로 사라지지 않게 하기 위해서다.
"""
import math

import torch


class MasterAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def load_state_dict(self, state_dict):
        """기본 구현은 상태 텐서를 파라미터 dtype(bf16)으로 바꾸므로, 저장된 fp32 값을 다시 넣는다."""
        super().load_state_dict(state_dict)
        saved = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for i, p in zip(saved, params):
            for key in ("master", "exp_avg", "exp_avg_sq"):
                if key in state_dict["state"].get(i, {}):
                    self.state[p][key] = state_dict["state"][i][key].to(device=p.device, dtype=torch.float32, copy=True)

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["master"] = p.detach().float().clone()
                    st["exp_avg"] = torch.zeros_like(st["master"])
                    st["exp_avg_sq"] = torch.zeros_like(st["master"])
                master, m, v = st["master"], st["exp_avg"], st["exp_avg_sq"]
                grad = p.grad.float()
                st["step"] += 1
                t = st["step"]
                master.mul_(1 - lr * wd)
                m.lerp_(grad, 1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
                master.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
                p.copy_(master)
