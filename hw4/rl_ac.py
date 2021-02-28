import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_pg import TrainBatch, PolicyAgent, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # TODO:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        hidden_layers = kw['hidden_layers']
        bias = kw['n_bias']
        base_layers = list()

        base_layers.append(nn.Linear(in_features, hidden_layers[0], bias=bias))
        base_layers.append(nn.ReLU())
        for h1, h2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            base_layers.append(nn.Linear(h1, h2, bias=bias))
            base_layers.append(nn.ReLU())
        self.base = nn.Sequential(*base_layers)
        self.action_layer = nn.Linear(hidden_layers[-1], out_actions)
        self.value_layer = nn.Linear(hidden_layers[-1], 1)
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # TODO:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        state = self.base(x)
        state_values = self.value_layer(state)
        action_scores = self.action_layer(state)
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device="cpu", **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        in_features = env.observation_space.shape[0]
        out_actions = env.action_space.n
        net = AACPolicyNet(in_features, out_actions, **kw)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):
    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        actions, _ = self.p_net(self.curr_state)
        actions_proba = torch.softmax(actions, dim=0)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        assert len(batch) == len(action_scores) == len(state_values)
        state_values = state_values.view(-1)
        advantage = self._policy_weight(batch, state_values)
        loss_p = self._policy_loss(batch, action_scores, advantage)
        loss_v = self._value_loss(batch, state_values)
        # ========================
        #print(loss_v.item(), (loss_p / (loss_v * self.delta)).item())
        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return (
            loss_t,
            dict(
                loss_p=loss_p.item(),
                loss_v=loss_v.item(),
                adv_m=advantage.mean().item(),
            ),
        )

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        rewards = batch.q_vals
        #rewards = (rewards - rewards.mean()) / (rewards.std())
        advantage = rewards - state_values.detach()
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        rewards = batch.q_vals
        #rewards = (rewards - rewards.mean()) / (rewards.std())
        loss_v = torch.nn.functional.mse_loss(state_values, rewards)
        # ========================
        return loss_v
