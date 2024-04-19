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
