import torch
import torch.nn as nn
from torchdiffeq import odeint
from utils import utils


class DiffeqSolverDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, ode_func, method='dopri5', odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DiffeqSolverDecoder, self).__init__()
        self.diffeq_solver = DiffeqSolver(ode_func, method, odeint_rtol, odeint_atol)
        self.output_layer = nn.Sequential(nn.Linear(input_dim, output_dim))
        utils.init_network_weights(self.output_layer)

    def forward(self, first_point, time_steps_to_predict):
        sol_z = self.diffeq_solver(first_point, time_steps_to_predict)
        pred_x = self.output_layer(sol_z)
        return sol_z, pred_x


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DiffeqSolver, self).__init__()
        self.ode_func = ode_func
        self.method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        sol_z = odeint(self.ode_func, first_point, time_steps_to_predict, rtol=self.odeint_rtol, atol=self.odeint_atol,
                       method=self.method)
        sol_z = sol_z.permute(1, 2, 0, 3)
        return sol_z


class ODEFunc(nn.Module):
    def __init__(self, dim, n_layers, n_units, nonlinear=nn.Tanh):
        super(ODEFunc, self).__init__()
        layers = [nn.Linear(dim, n_units)]
        for _ in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, t, y):
        return self.layers(y)


class RNNDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, n_units, nonlinear=nn.Tanh):
        super(RNNDecoder, self).__init__()
        self.rnn_cell = nn.GRUCell(output_dim + 1, latent_dim)
        self.output_net = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nonlinear(),
            nn.Linear(n_units, output_dim)
        )
        utils.init_network_weights(self.output_net)

    def forward(self, first_point, time_steps_to_predict):
        # first_point: [n_traj, B, L]
        _, batch_size, _ = first_point.size()
        device = utils.get_device(time_steps_to_predict)

        zero_delta_t = torch.Tensor([0.]).to(device)
        delta_ts = torch.cat((zero_delta_t, time_steps_to_predict[1:] - time_steps_to_predict[:-1]))
        delta_ts = delta_ts.repeat(batch_size, 1)  # [B, T]

        hidden_state = first_point.squeeze(0)
        sol_z = [hidden_state]  # [B, L]
        pred_x = [self.output_net(hidden_state)]  # [B, N]
        for t in range(1, time_steps_to_predict.shape[0]):
            rnn_input = torch.cat((pred_x[t - 1], delta_ts[:, t].unsqueeze(-1)), dim=-1)
            hidden_state = self.rnn_cell(rnn_input, hidden_state)
            pred = self.output_net(hidden_state)

            sol_z.append(hidden_state)
            pred_x.append(pred)
        sol_z = torch.stack(sol_z, dim=0)  # [T, B, L]
        sol_z = sol_z.permute(1, 0, 2).unsqueeze(0)  # [1, B, T, L]
        pred_x = torch.stack(pred_x, dim=0)  # [T, B, N]
        pred_x = pred_x.permute(1, 0, 2).unsqueeze(0)  # [1, B, T, N]
        return sol_z, pred_x
