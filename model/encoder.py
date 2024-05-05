from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU

try:
    from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.modules.mamba_simple import Block, Mamba
    from mamba_ssm.utils.generation import GenerationMixin
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
except ImportError:
    create_block, _init_weights = None, None
    MambaConfig = None
    Block, Mamba = None, None
    GenerationMixin = None
    load_config_hf, load_state_dict_hf = None, None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from utils.utils import get_device
from utils import utils


class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=100,
                 device=torch.device("cpu")):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            utils.init_network_weights(self.update_gate)
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            utils.init_network_weights(self.reset_gate)
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2))
            utils.init_network_weights(self.new_state_net)
        else:
            self.new_state_net = new_state_net

    def forward(self, y_mean, y_std, x, masked_update=True):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert (not torch.isnan(new_y).any())

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the
            # current time point
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            utils.check_mask(x[:, :, :n_data_dims], mask)

            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

            assert (not torch.isnan(mask).any())

            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std


class mamba_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100, new_state_net=None, d_state=16, d_conv=4, expand=2):
        super(mamba_unit, self).__init__()
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mamba = Mamba(input_dim + latent_dim * 2, d_state=d_state, d_conv=d_conv, expand=expand)
        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(input_dim + latent_dim * 2, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2)
            )
            utils.init_network_weights(self.new_state_net)
        else:
            self.new_state_net = new_state_net

    def forward(self, y_mean, y_std, x, conv_state=None, ssm_state=None, masked_update=True):
        """
        mamba update
        Args:
            y_mean: [1, B, L]
            y_std: [1, B, L]
            x: [1, B, input_dim]
            conv_state: [B, input_dim * expand, d_conv]
            ssm_state: [B, input_dim * expand, d_state]
            masked_update: bool

        Returns:

        """
        # naive update
        y_concat = torch.cat([y_mean, y_std, x], -1)  # [1, B, rec * 2 + input]
        y_concat = y_concat.permute(1, 0, 2)  # [B, 1, rec * 2 + input]
        y_concat = self.mamba(y_concat)
        y_concat = y_concat.permute(1, 0, 2)
        new_y, new_y_std = utils.split_last_dim(self.new_state_net(y_concat))

        # batch_size = x.size(1)
        # mamba_input = x.permute(1, 0, 2)  # [B, 1, input]
        # mamba_out, new_conv_state, new_ssm_state = self.mamba.step(mamba_input, conv_state, ssm_state)
        # new_y, new_y_std = utils.split_last_dim(self.new_state_net(mamba_out.squeeze(1)))
        # new_y = new_y.unsqueeze(0)
        # new_y_std = new_y_std.unsqueeze(0)

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the
            # current time point
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            utils.check_mask(x[:, :, :n_data_dims], mask)

            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()  # [1, B, 1]
            mamba_mask = mask.squeeze(0).unsqueeze(-1)  # [B, 1, 1]

            assert (not torch.isnan(mask).any())

            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std
            # new_conv_state = mamba_mask * new_conv_state + (1 - mamba_mask) * conv_state
            # new_ssm_state = mamba_mask * new_ssm_state + (1 - mamba_mask) * ssm_state

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std


class LSTM_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(LSTM_unit, self).__init__()
        self.forget_gate = nn.Sequential(
            nn.Linear(input_dim + latent_dim * 2, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        utils.init_network_weights(self.forget_gate)

        self.input_gate = nn.Sequential(
            nn.Linear(input_dim + latent_dim * 2, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid()
        )
        utils.init_network_weights(self.input_gate)

        self.Gate_gate = nn.Sequential(
            nn.Linear(input_dim + latent_dim * 2, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Tanh()
        )
        utils.init_network_weights(self.Gate_gate)

        self.output_gate = nn.Sequential(
            nn.Linear(input_dim + latent_dim * 2, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2),
        )
        utils.init_network_weights(self.output_gate)

    def forward(self, y_mean, y_std, x, cell, masked_update=True):
        y_concat = torch.cat([y_mean, y_std, x], -1)
        f_t = self.forget_gate(y_concat)
        cell_prime = f_t * cell

        i_t = self.input_gate(y_concat)
        C_t = self.Gate_gate(y_concat)
        new_cell = cell_prime + i_t * C_t

        o_t = self.output_gate(y_concat)
        new_y, new_y_std = utils.split_last_dim(o_t)
        new_y = new_y * F.tanh(new_cell)
        new_y_std = new_y_std * F.tanh(new_cell)

        assert (not torch.isnan(new_y).any())
        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the
            # current time point
            n_data_dims = x.size(-1) // 2
            mask = x[:, :, n_data_dims:]
            utils.check_mask(x[:, :, :n_data_dims], mask)

            mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

            assert (not torch.isnan(mask).any())

            new_y = mask * new_y + (1 - mask) * y_mean
            new_y_std = mask * new_y_std + (1 - mask) * y_std
            new_cell = mask * new_cell + (1 - mask) * cell

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std, new_cell


class Encoder_z0_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards
    # from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE running
    # from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None,
                 z0_dim=None, GRU_update=None,
                 n_gru_units=100,
                 use_lstm=False,
                 use_mamba=False,
                 device=torch.device("cpu")):

        super(Encoder_z0_ODE_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim,
                                       n_units=n_gru_units,
                                       device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None
        self.use_lstm = use_lstm
        self.use_mamba = use_mamba

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2), )
        utils.init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True, save_info=False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:, 0, :].unsqueeze(0)

            if self.use_lstm:
                prev_cell = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
                last_yi, last_yi_std, _ = self.GRU_update(prev_y, prev_std, xi, prev_cell)
            elif self.use_mamba:
                prev_conv_state = torch.zeros(n_traj, n_dims * self.GRU_update.expand, self.GRU_update.d_conv).to(self.device)
                prev_ssm_state = torch.zeros(n_traj, n_dims * self.GRU_update.expand, self.GRU_update.d_state).to(self.device)
                last_yi, last_yi_std, _, _ = self.GRU_update(prev_y, prev_std, xi, prev_conv_state, prev_ssm_state)
            else:
                last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:

            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards=run_backwards,
                save_info=save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = utils.split_last_dim(self.transform_z0(torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        std_z0 = torch.clamp(std_z0, min=1e-20)
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps, run_backwards=True, save_info=False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        t0 = time_steps[0]
        if run_backwards:
            t0 = time_steps[-1]

        device = get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_cell = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        if self.use_mamba:
            prev_conv_state = torch.zeros(n_traj, n_dims * self.GRU_update.expand, self.GRU_update.d_conv).to(device)
            prev_ssm_state = torch.zeros(n_traj, n_dims * self.GRU_update.expand, self.GRU_update.d_state).to(device)
        else:
            prev_conv_state, prev_ssm_state = None, None

        prev_t, t_i = t0 - 0.01, t0
        if run_backwards:
            prev_t, t_i = time_steps[-1] + 0.01, time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        # print("minimum step: {}".format(minimum_step))

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_steps_len = len(time_steps)
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            # dt太小直接用f'*dt算
            if abs(t_i - prev_t) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert (not torch.isnan(ode_sol).any())
            # 用ode solver感觉意义不大啊，dt一般都很小
            else:
                n_intermediate_tp = max(2, (abs(t_i - prev_t) / minimum_step).int())

                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert (not torch.isnan(ode_sol).any())

            # 不会是diffeq库的问题吧
            if torch.mean(ode_sol[:, :, 0, :] - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_y))
                exit()
            # assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)

            if self.use_lstm:
                yi, yi_std, cell = self.GRU_update(yi_ode, prev_std, xi, prev_cell)
                prev_y, prev_std, prev_cell = yi, yi_std, cell
            elif self.use_mamba:
                yi, yi_std, conv_state, ssm_state = self.GRU_update(
                    yi_ode, prev_std, xi, prev_conv_state, prev_ssm_state
                )
                prev_y, prev_std, prev_conv_state, prev_ssm_state = yi, yi_std, conv_state, ssm_state
            else:
                yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
                prev_y, prev_std = yi, yi_std

            prev_t, t_i = time_steps[i], time_steps[(i + 1) % time_steps_len]
            if run_backwards:
                prev_t, t_i = time_steps[i], time_steps[i - 1]

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(),  # "yi_from_data": yi_from_data,
                     "yi": yi.detach(), "yi_std": yi_std.detach(),
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert (not torch.isnan(yi).any())
        assert (not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys, extra_info


class Encoder_z0_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, lstm_output_size=20,
                 use_delta_t=True, device=torch.device("cpu")):

        super(Encoder_z0_RNN, self).__init__()

        self.gru_rnn_output_size = lstm_output_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.use_delta_t = use_delta_t

        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.gru_rnn_output_size, 50),
            nn.Tanh(),
            nn.Linear(50, latent_dim * 2), )

        utils.init_network_weights(self.hiddens_to_z0)

        input_dim = self.input_dim

        if use_delta_t:
            self.input_dim += 1
        self.gru_rnn = GRU(self.input_dim, self.gru_rnn_output_size).to(device)

    def forward(self, data, time_steps, run_backwards=True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        # data shape: [n_traj, n_tp, n_dims]
        # shape required for rnn: (seq_len, batch, input_size)
        # t0: not used here
        n_traj = data.size(0)

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        data = data.permute(1, 0, 2)

        if run_backwards:
            # Look at data in the reverse order: from later points to the first
            data = utils.reverse(data)

        if self.use_delta_t:
            delta_t = time_steps[1:] - time_steps[:-1]
            if run_backwards:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            # append zero delta t in the end
            delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1, n_traj)).unsqueeze(-1)
            data = torch.cat((delta_t, data), -1)

        outputs, _ = self.gru_rnn(data)

        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)
        last_output = outputs[-1]

        self.extra_info = {"rnn_outputs": outputs, "time_points": time_steps}

        mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
        std = std.abs()
        std = torch.clamp(std, min=1e-20)

        assert (not torch.isnan(mean).any())
        assert (not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0)


class EncoderAttention(nn.Module):
    def __init__(self, input_dim, d_model, nhead, d_ff, num_layers, latent_dim, max_length,
                 dropout=0.1, use_split=False):
        super(EncoderAttention, self).__init__()
        self.position_encoder = PositionEncoder(input_dim)
        self.num_layers = num_layers
        self.attention_encoder_layers = nn.ModuleList(
            [EncoderAttentionLayer(input_dim, d_model, nhead, d_ff, dropout=dropout, use_split=use_split)
             for _ in range(num_layers)])
        self.norm = Norm(input_dim)
        self.max_length = max_length
        # self.transform_tp = nn.Sequential(
        #     nn.Linear(max_length, max_length // 2),
        #     nn.Tanh(),
        #     nn.Linear(max_length // 2, 1),
        # )
        # initialize transform tp to mean.
        self.transform_tp = nn.Linear(max_length, 1)
        self.transform_tp.weight.data.fill_(1 / max_length)
        self.transform_tp.bias.data.fill_(0)

        self.transform_z0 = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, latent_dim * 2), )
        utils.init_network_weights(self.transform_z0)

    def forward(self, x, time_steps, run_backwards=True):
        """
        :param x: [batch_size, n_tp, input_dim * 2]
        :param time_steps: [n_tp]
        :param run_backwards:
        :return:
        """
        device = utils.get_device(time_steps)
        batch_size, n_tp, input_dim = x.size()
        n_data_dims = input_dim // 2
        mask = x[:, :, n_data_dims:]
        utils.check_mask(x[:, :, :n_data_dims], mask)
        mask = mask.clone()
        x = x[:, :, :n_data_dims]

        x = self.position_encoder(x, mask, time_steps)

        # assert self.max_length >= n_tp, f"max_length {self.max_length} must be larger than or equal n_tp {n_tp}"
        # # padding
        # padding = torch.zeros(batch_size, self.max_length - n_tp, n_data_dims).to(device)
        # x = torch.cat((x, padding), dim=1)
        # mask = torch.cat((mask, padding), dim=1)

        for layer in self.attention_encoder_layers:
            x = layer(x, mask)
        x = self.norm(x)  # shape: [batch_size, n_tp, input_dim]

        # Find the mean on all tp.
        # if run_backwards:
        #     x = x[:, 0, :]
        # else:
        x = torch.mean(x, 1)  # [batch_size, input_dim]
        # x = x.permute(0, 2, 1)  # [batch_size, input_dim, n_tp]
        # x = self.transform_tp(x)  # [batch_size, input_dim, 1]
        # x = x.squeeze(-1)

        x = x.unsqueeze(0)
        x = self.transform_z0(x)
        mean_z0, std_z0 = utils.split_last_dim(x)
        std_z0 = std_z0.abs()
        std_z0 = torch.clamp(std_z0, min=1e-20)

        return mean_z0, std_z0


class PositionEncoder(nn.Module):
    def __init__(self, input_dim):
        super(PositionEncoder, self).__init__()
        self.input_dim = input_dim

    def forward(self, data, mask, time_steps):
        batch_size, n_tp, n_dim = data.size()
        device = utils.get_device(time_steps)

        # make embeddings relatively larger
        data = data * math.sqrt(self.input_dim)

        pe = torch.zeros(n_tp, n_dim, requires_grad=False).to(get_device(data))
        time_steps = time_steps.unsqueeze(1)
        # original pe
        steps = torch.arange(n_tp, device=device)
        steps = steps.unsqueeze(1)

        div_term_0 = torch.exp(
            torch.arange(0, self.input_dim, 2, device=device)
            * -(math.log(10000.0) / self.input_dim)
        )
        div_term_1 = torch.exp(
            torch.arange(1, self.input_dim, 2, device=device)
            * -(math.log(10000.0) / self.input_dim)
        )

        pe[:, 0::2] = torch.sin(steps * div_term_0)
        pe[:, 1::2] = torch.cos(steps * div_term_1)

        pe = pe.unsqueeze(0)

        # only mask the tp when no observation is taken.
        # pe = pe.repeat(batch_size, 1, 1)
        # mask = mask.sum(-1, keepdim=True) == 0.
        # mask = mask.repeat(1, 1, n_dim)
        # pe = torch.masked_fill(pe, mask, 0)

        data = data + pe
        # subsample_num = 800
        # if time_steps.size(0) > subsample_num:
        #     subsample_idx = torch.randperm(time_steps.size(0))[:subsample_num].sort()[0]
        #     data = data[:, subsample_idx, :]
        #     mask = mask[:, subsample_idx, :]
        return data


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dropout=0.1, use_split=False):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.WQs = nn.Linear(input_dim, d_model * nhead)
        self.WKs = nn.Linear(input_dim, d_model * nhead)
        self.WVs = nn.Linear(input_dim, d_model * nhead)
        self.WO = nn.Linear(d_model * nhead, input_dim)
        self.n_tp_split = 192 if use_split else None

    def forward(self, data, mask):
        # mask = mask.sum(dim=-1, keepdim=True) == 0.
        # mask = mask.repeat(1, 1, self.input_dim)
        # data = torch.masked_fill(data, mask, 0)

        batch_size, n_tp, input_dim = data.size()
        Qs = self.WQs(data)
        Ks = self.WKs(data)
        Vs = self.WVs(data)

        Qs = Qs.reshape(batch_size, n_tp, self.d_model, self.nhead)
        Ks = Ks.reshape(batch_size, n_tp, self.d_model, self.nhead)
        Vs = Vs.reshape(batch_size, n_tp, self.d_model, self.nhead)

        Qs = Qs.permute(0, 3, 1, 2)  # [batch_size, nhead, n_tp, d_model]
        Ks = Ks.permute(0, 3, 2, 1)  # [batch_size, nhead, d_model, n_tp]
        Vs = Vs.permute(0, 3, 1, 2)  # [batch_size, nhead, n_tp, d_model]

        # Not care when n_tp_split is not None.
        if self.n_tp_split is not None:
            scores = []
            for i in range(0, n_tp, self.n_tp_split):
                end = i + self.n_tp_split
                if end > n_tp:
                    end = n_tp
                Qs_split = Qs[:, :, i:end, :]
                Ks_split = Ks[:, :, i:end, :]
                Vs_split = Vs[:, :, i:end, :]
                scores_split = torch.matmul(Qs_split, Ks_split.permute(0, 1, 3, 2)) / math.sqrt(self.d_model)
                scores_split = F.softmax(scores_split, dim=-1)
                scores_split = torch.matmul(scores_split, Vs_split)
                scores.append(scores_split)
            scores = torch.cat(scores, dim=-2)
        else:
            # shape: [batch_size, nhead, n_tp, n_tp]
            scores = torch.matmul(Qs, Ks) / math.sqrt(self.d_model)
            if mask is not None:
                # shape: [batch_size, n_tp]. True indicates that there are no observation in the time_point.
                mask = torch.sum(mask, dim=-1) == 0.
                scores = scores.permute(1, 2, 0, 3)
                # make these time points' score very low.
                scores = scores.masked_fill(mask, -1e9)
                scores = scores.permute(2, 0, 1, 3)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)
            scores = torch.matmul(scores, Vs)

        scores = scores.permute(0, 2, 1, 3)
        scores = scores.reshape(batch_size, n_tp, self.nhead * self.d_model)
        scores = self.WO(scores)
        return scores


class FeedForward(nn.Module):
    def __init__(self, input_dim, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(input_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, input_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super(Norm, self).__init__()

        self.size = input_dim
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        x = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return x


class EncoderAttentionLayer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, d_ff, dropout=0.1, use_split=False):
        super(EncoderAttentionLayer, self).__init__()
        self.norm1 = Norm(input_dim, eps=1e-6)
        self.norm2 = Norm(input_dim, eps=1e-6)
        self.attn = MultiHeadAttention(input_dim, d_model, nhead, dropout=dropout, use_split=use_split)
        self.feedforward = FeedForward(input_dim, d_ff, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        x = x + self.dropout2(self.feedforward(self.norm2(x)))
        return x


class EncoderMamba(nn.Module):
    def __init__(self, input_dim, latent_dim, max_length, d_state=16, d_conv=4, expand=2):
        super(EncoderMamba, self).__init__()
        self.mamba = Mamba(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mixer_model = MixerModel(
            d_model=input_dim,
            n_layer=12,
            ssm_cfg=None,
            rms_norm=True,
            fused_add_norm=True,
            residual_in_fp32=True
        )
        self.position_encoder = PositionEncoder(input_dim)
        self.max_length = max_length
        # self.transform_tp = nn.Sequential(
        #     nn.Linear(max_length, max_length // 2),
        #     nn.Tanh(),
        #     nn.Linear(max_length // 2, 1),
        # )
        self.transform_tp = nn.Linear(max_length, 1)
        self.transform_tp.weight.data.fill_(1 / max_length)
        self.transform_tp.bias.data.fill_(0)

        self.transform_z0 = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, latent_dim * 2), )
        utils.init_network_weights(self.transform_z0)

    def forward(self, x, time_steps, run_backwards=True):
        device = utils.get_device(time_steps)
        batch_size, n_tp, input_dim = x.size()
        n_data_dims = input_dim // 2
        mask = x[:, :, n_data_dims:]
        utils.check_mask(x[:, :, :n_data_dims], mask)
        mask = mask.clone()
        x = x[:, :, :n_data_dims]
        # TODO: How to use the mask?

        x = self.position_encoder(x, mask, time_steps)

        # # padding
        # assert self.max_length >= n_tp, f"max_length {self.max_length} must be larger than or equal n_tp {n_tp}"
        # padding = torch.zeros(batch_size, self.max_length - n_tp, n_data_dims).to(device)
        # x = torch.cat((x, padding), dim=1)
        # mask = torch.cat((mask, padding), dim=1)

        # Dont't use backwards.
        # if run_backwards:
        #     x = x.flip(1)
        # x = self.mamba(x)
        x = self.mixer_model(x)  # [batch_size, n_tp, input_dim]

        # mask = mask.sum(dim=-1, keepdim=True) == 0.
        # mask = mask.repeat(1, 1, n_data_dims)
        # x = torch.masked_fill(x, mask, 0)

        x = x.mean(1)
        # x = x.permute(0, 2, 1)  # [batch_size, input_dim, n_tp]
        # x = self.transform_tp(x)  # [batch_size, input_dim, 1]
        # x = x.squeeze(-1)

        x = x.unsqueeze(0)
        x = self.transform_z0(x)
        mean_z0, std_z0 = utils.split_last_dim(x)
        std_z0 = std_z0.abs()
        std_z0 = torch.clamp(std_z0, min=1e-20)

        return mean_z0, std_z0


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, inference_params=None):
        residual = None
        for layer in self.layers:
            x, residual = layer(
                x, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (x + residual) if residual is not None else x
            x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            x = fused_add_norm_fn(
                x,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return x
