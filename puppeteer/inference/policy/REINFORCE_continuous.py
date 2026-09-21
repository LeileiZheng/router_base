import torch
import atexit
import os
import datetime
import json
import numpy as np
import torch.nn as nn
import yaml
import logging
from utils.other_utils import Singleton
from inference.policy.base_policy import LLMPolicy, LearningPolicy
from model.embedding import RewardModelTokenRepresentation

global_config = yaml.safe_load(open("puppeteer/config/global.yaml", "r"))
logger = logging.getLogger("train")

@Singleton
class MLP_PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 32)
        self.fc4 = torch.nn.Linear(32, output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


@Singleton
class ContinuousREINFORCE(LearningPolicy):
    def __init__(self, agent_graph, action_graph, config_path="puppeteer/config/policy.json"):
        super().__init__(agent_graph, action_graph)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set parameters from config
        self.device = self.config["device"]["type"]
        
        # Training parameters
        self.model_path = self.config["paths"]["model_path"]
        configured_training = bool(self.config["training"].get("training", True))
        dataset_mode = self.config.get("dataset_mode")
        # validation/test are evaluation modes in the CLI.  Keep this guard in
        # the policy as well as main.py so a stale config cannot enable updates.
        self.training = configured_training and dataset_mode not in {"validation", "test"}
        self.loading = self.config["training"]["loading"]
        self.learning_rate = self.config["training"]["learning_rate"]
        self.gamma = self.config["training"]["gamma"]
        self.sample_size = self.config["training"]["sample_size"]
        self.lambda_kl_loss = self.config["training"]["lambda_kl_loss"]
        self.entropy_coef = float(self.config["training"]["entropy_coef"])
        if self.entropy_coef < 0:
            raise ValueError("training.entropy_coef must be non-negative")

        # Sequential base reward: task correctness minus a uniform cost for
        # every router-selected reasoning-agent call.  These values are kept
        # independent of token/model accounting, which remains available only
        # as execution metadata.
        reward_config = self.config["reward"]
        self.task_reward_correct = float(reward_config["task_reward_correct"])
        self.task_reward_incorrect = float(reward_config["task_reward_incorrect"])
        self.call_cost = float(reward_config["call_cost"])
        if self.call_cost < 0:
            raise ValueError("reward.call_cost must be non-negative")

        # Legacy parallel-selection parameters. Sequential forward always samples
        # exactly one categorical action and does not use these values.
        self.max_num_agents = self.config["agent"]["max_num_agents"] 
        self.next_num_agents = self.config["agent"]["next_num_agents"] 
        self.max_path = self.config["agent"]["max_path"]
        self.threshold = self.config["agent"]["threshold"]
        
        # LLM parameters
        self.llm_prior = self.config["llm"]["prior"]
        self.llm_prior_redistribution = self.config["llm"]["prior_redistribution"]
        self.redistribution_weight = self.config["llm"]["redistribution_weight"]
        
        # Initialize state representation and policy network
        self.state_representation = RewardModelTokenRepresentation()
        self.policy_network = MLP_PolicyNetwork(self.state_representation.dim, self.actions_dim) 
        self.policy_network = self.policy_network.to(self.device) 
        self.policy_network.train(self.training)
        if not self.training:
            self.load_model(self.get_latest_model_path())
        if self.loading:
            self.load_model(self.model_path)

        # Agent setup
        self.agent_hash_list = agent_graph.hash_nodes
        self.agent_role_list = agent_graph.role_nodes
        
        # Initialize tracking variables
        self.executed_trajectories = []
        self.execution_count = 0 
        self.current_trajectories = []
        self.current_trajectory_idx = 0
        
        self.policy_losses = []
        self.rewards_history = []
        self.action_probs_history = []
        self.llm_action_probs_history = []
        self.reward_from_rm = []
        self.accumulated_acc = []

        # Setup actions
        self.end_action = torch.tensor(self.agent_graph.terminator_agent_index, device=self.device)
        # self.web_actions = torch.tensor(self.agent_graph.search_agent_indices, device=self.device)

        self.current_task = None
        # Legacy task-diff based lifecycle tracking. Query starts are now
        # identified by global_info.path_id == -1 in forward().
        # self.previous_task = None
        self.global_step = 0    
        self.prob_step=0
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.max_step_num = global_config.get("graph").get("max_step_num")
        self.llm_policy = LLMPolicy(self.agent_graph, self.action_graph)
        
        atexit.register(self.save_model)

    def save_model(self, path=None, tag=None):
        """Save model with config"""
        path = self.config["paths"]["checkpoint_path"]

        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'policy_net_{timestamp}' + (f'_{tag}' if tag else '') + '.pt'
        save_path = os.path.join(path, filename)
        
        checkpoint = {
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'input_dim': self.policy_network.input_dim,
            'output_dim': self.policy_network.output_dim,
            'timestamp': timestamp,
            'config': self.config,  # Save the config with the model
            'metadata': {
                'tag': tag,
                'version': '1.0',
            }
        }
        
        try:
            torch.save(checkpoint, save_path)
            print(f"Model saved successfully to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return None

    def update_executed_trajectories(self):
        self.execution_count += 1
        self.executed_trajectories.append([])
        self.current_trajectories = self.executed_trajectories[-1]
    
    def select_single_agent(self, action_probs):
        """Select exactly one routing action from the policy distribution."""
        dist = torch.distributions.Categorical(probs=action_probs)
        if self.training:
            agent_idx = dist.sample()
        else:
            agent_idx = torch.argmax(action_probs, dim=-1)
        return agent_idx, dist


    def append_to_trajectory(self, trajectory_idx, agent_idx, prob_value, global_info, prior_action_probs, m, rew=0, entropy=None):
        is_terminator = agent_idx.item() == self.end_action.item()
        step_reward = torch.tensor(
            0.0 if is_terminator else -self.call_cost,
            dtype=torch.float32,
            device=self.device,
        )
        transition = {
            'prob': prob_value,
            'log_prob': m.log_prob(agent_idx),
            'state_identifier': global_info.workflow.state,
            'action': self.agent_role_list[agent_idx.item()],
            'reward': step_reward,
            'is_reasoning_call': not is_terminator,
            'reward_model': rew,
            'prior_prob': prior_action_probs[agent_idx.item()] if prior_action_probs is not None else None
        }
        if entropy is not None:
            transition['entropy'] = entropy
        self.current_trajectories[trajectory_idx].append(transition)
        print(trajectory_idx, self.current_trajectories[trajectory_idx])

    def forward(self, global_info):
        is_initial_step = global_info.path_id == -1
        phase = "Init" if is_initial_step else "Following"
        print("\033[1;33m{} Policy Forward\033[0m".format(phase))
        logger.info("[{} Policy Forward]".format(phase))

        self.current_task = global_info.task
        if is_initial_step:
            self.update_executed_trajectories()

        state, rew = self.get_state_representation(global_info)
        with torch.set_grad_enabled(self.training):
            action_probs = self.policy_network(state)

            # An empty trajectory has no answer to score.  Mask STOP before
            # constructing the distribution so sampling, log_prob, and entropy
            # all use exactly the same probabilities.
            if is_initial_step:
                action_probs = action_probs.clone()
                action_probs[..., self.end_action.item()] = 0.0
                probability_mass = action_probs.sum(dim=-1, keepdim=True)
                if torch.any(probability_mass <= 0):
                    raise ValueError("Initial action distribution has no non-STOP probability mass")
                action_probs = action_probs / probability_mass

            agent_idx, dist = self.select_single_agent(action_probs)
            entropy = dist.entropy().squeeze(0) if self.training else None

        self.action_probs_history.append(action_probs.T.squeeze(1))
        self.reward_from_rm.append(rew)
        logger.info("Action probs: {}".format(action_probs))

        assert agent_idx.numel() == 1, "Sequential routing must select one agent"

        self.current_trajectory_idx = 0
        if len(self.current_trajectories) == 0:
            self.current_trajectories.append([])
        assert len(self.current_trajectories) == 1, "Sequential routing must keep one trajectory"

        prob_value = action_probs[0, agent_idx.item()]
        self.append_to_trajectory(
            self.current_trajectory_idx,
            agent_idx,
            prob_value,
            global_info,
            None,
            dist,
            rew,
            entropy,
        )

        agent_indices = agent_idx.reshape(-1)
        assert agent_indices.numel() == 1, "Sequential policy.forward() must return one agent"
        print("Agent Indices: {}".format(agent_indices))
        selected_agents = [self.agent_hash_list[index.item()] for index in agent_indices]
        assert len(selected_agents) == 1, "Sequential policy.forward() must return one agent"
        return selected_agents
    
    def calculate_returns(self, trajectory):
        returns = []
        # A max-step termination is an environment event, not a policy action.
        # Seed the return with its terminal reward so it keeps the same temporal
        # position as the legacy appended Terminator transition without adding
        # a fictitious log_prob-bearing action to the trajectory.
        R = trajectory[-1].get('terminal_reward', 0) if trajectory else 0
        for t in reversed(trajectory):
            R = t.get('reward', 0) + self.gamma * R 
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)
    
    def get_state_representation(self, global_info):
        role_list = global_info.agent_role_list()
        print(role_list)
        state_context = self.agent_graph.get_agent_dialog_history(role_list, question=global_info.task.get("Question"))
        print(state_context)
        print(type(state_context))
        state, reward = self.state_representation(state_context)
        print(state, reward)
        return state, reward        

    def _reset_batch_state(self):
        """Release all graph-bearing tensors after an update or eval episode."""
        self.current_trajectories = []
        self.executed_trajectories = []
        self.execution_count = 0
        self.reward_from_rm = []
        self.action_probs_history = []
        self.llm_action_probs_history = []

    def update(self):
        logger.info("Update")
        logger.info("Executed trajectories: {}".format(self.executed_trajectories))

        if not self.training:
            # Evaluation uses greedy selection and never constructs or consumes
            # graph-bearing entropy records.
            metrics = {}
            if self.action_probs_history:
                detached_probs = [probs.detach() for probs in self.action_probs_history]
                metrics['reasoning/action_probs'] = torch.sum(
                    torch.stack(detached_probs), dim=0
                )
                metrics['evaluation/entropy'] = torch.stack([
                    torch.distributions.Categorical(probs=probs).entropy()
                    for probs in detached_probs
                ]).mean().item()
            logger.info("metrics: {}".format(metrics))
            self._reset_batch_state()
            return {}

        # Preserve the existing full-batch update gate.  Once the configured
        # number of episode slots exists, normalize by the number that actually
        # contains one valid finalized sequential trajectory.
        if len(self.executed_trajectories) < self.sample_size:
            return {}

        batch_episodes = self.executed_trajectories[:self.sample_size]
        valid_trajectories = []
        for episode in batch_episodes:
            episode_trajectories = [
                trajectory
                for trajectory in episode
                if trajectory
                and trajectory[-1].get('finalized', False)
                and all(
                    item.get('log_prob') is not None and item.get('entropy') is not None
                    for item in trajectory
                )
            ]
            if len(episode_trajectories) > 1:
                raise ValueError("Sequential routing permits one valid trajectory per episode")
            if episode_trajectories:
                valid_trajectories.append(episode_trajectories[0])

        effective_batch_size = len(valid_trajectories)
        if effective_batch_size == 0:
            logger.warning("Skipping update(): batch has no valid finalized trajectories")
            self._reset_batch_state()
            return {}

        trajectory_pg_losses = []
        trajectory_kl_losses = []
        trajectory_entropies = []
        episode_returns = []
        episode_lengths = []
        episode_last_rewards = []
        episode_acc = []
        episode_tokens = []
        episode_cost = []
        episode_metrics = {}

        for trajectory in valid_trajectories:
            logger.info("Trajectory: {}".format(trajectory))
            returns = self.calculate_returns(trajectory)
            print("returns: {}".format(returns))
            logger.info("Trajectory returns: {}".format(returns))

            step_pg_losses = []
            step_kl_losses = []
            for transition, step_return in zip(trajectory, returns):
                step_pg_losses.append((-transition['log_prob'] * step_return).sum())
                if transition.get('prob') is not None and transition.get('prior_prob') is not None:
                    kl_value = transition['prior_prob'] * torch.log(
                        transition['prior_prob'] / (transition['prob'] + 1e-10)
                    )
                    # Preserve the legacy detached KL behavior.  Its default
                    # coefficient remains zero and its definition is outside
                    # the scope of the entropy refactor.
                    kl_value = kl_value.detach().clone().to(self.device)
                else:
                    kl_value = torch.zeros((), device=self.device)
                step_kl_losses.append(kl_value.sum())

            trajectory_pg_losses.append(torch.stack(step_pg_losses).sum())
            trajectory_kl_losses.append(torch.stack(step_kl_losses).sum())
            trajectory_entropies.append(torch.stack([
                transition['entropy'].reshape(()) for transition in trajectory
            ]).mean())

            last_transition = trajectory[-1]
            last_reward = last_transition.get(
                'terminal_reward', last_transition.get('reward', 0)
            )
            episode_returns.append(returns.sum().detach())
            episode_lengths.append(len(trajectory))
            episode_last_rewards.append(torch.as_tensor(last_reward).detach())
            episode_acc.append(1 if float(torch.as_tensor(last_reward).item()) > 0 else 0)
            episode_tokens.append(last_transition.get('total_tokens', 0))
            episode_cost.append(last_transition.get('total_cost', 0))
            for key, value in last_transition.get('metrics', {}).items():
                episode_metrics.setdefault(key, []).append(value)

        policy_gradient_loss = torch.stack(trajectory_pg_losses).mean()
        mean_kl_loss = torch.stack(trajectory_kl_losses).mean()
        kl_regularization_loss = self.lambda_kl_loss * mean_kl_loss
        entropy_mean = torch.stack(trajectory_entropies).mean()
        entropy_regularization_loss = -self.entropy_coef * entropy_mean
        total_loss = (
            policy_gradient_loss
            + kl_regularization_loss
            + entropy_regularization_loss
        )

        logger.info("Policy-gradient loss: {}".format(policy_gradient_loss))
        logger.info("Mean trajectory entropy: {}".format(entropy_mean))
        logger.info("Entropy regularization loss: {}".format(entropy_regularization_loss))
        logger.info("Total loss: {}".format(total_loss))

        self.optimizer.zero_grad()
        total_loss.backward()
        squared_gradient_norm = torch.zeros((), device=self.device)
        for parameter in self.policy_network.parameters():
            if parameter.grad is not None:
                squared_gradient_norm += parameter.grad.detach().pow(2).sum()
        gradient_norm = squared_gradient_norm.sqrt()
        self.optimizer.step()

        metrics = {
            'reasoning/action_probs': torch.sum(
                torch.stack(self.action_probs_history), dim=0
            ),
            'reasoning/reward_from_rm': sum(self.reward_from_rm),
            'reasoning/acc': np.mean(episode_acc),
            'reasoning/tokens': np.mean(episode_tokens),
            'reasoning/cost': np.mean(episode_cost),
            'reasoning/mean_return': torch.stack(episode_returns).mean().item(),
            'reasoning/mean_last_reward': torch.stack(episode_last_rewards).float().mean().item(),
            'training/policy_loss': total_loss.item(),
            'training/policy_gradient_loss': policy_gradient_loss.item(),
            'training/mean_kl_loss': mean_kl_loss.item(),
            'training/kl_regularization_loss': float(torch.as_tensor(kl_regularization_loss).item()),
            'training/entropy_mean': entropy_mean.item(),
            'training/entropy_regularization_loss': entropy_regularization_loss.item(),
            'training/total_loss': total_loss.item(),
            'training/entropy_coef': self.entropy_coef,
            'training/effective_batch_size': effective_batch_size,
            'training/average_trajectory_length': float(np.mean(episode_lengths)),
            'training/gradient_norm': gradient_norm.item(),
        }
        metrics.update({
            f'reasoning/{key}': np.mean([
                torch.as_tensor(value).detach().cpu().item()
                for value in values
            ])
            for key, values in episode_metrics.items()
        })
        logger.info("metrics: {}".format(metrics))

        self.global_step += 1
        self.policy_losses.append(total_loss.item())
        result = {
            'policy_loss': total_loss.item(),
            'policy_gradient_loss': policy_gradient_loss.item(),
            'entropy_mean': entropy_mean.item(),
            'entropy_regularization_loss': entropy_regularization_loss.item(),
            'total_loss': total_loss.item(),
            'entropy_coef': self.entropy_coef,
            'effective_batch_size': effective_batch_size,
            'average_trajectory_length': float(np.mean(episode_lengths)),
            'gradient_norm': gradient_norm.item(),
            'mean_reward': torch.stack(episode_returns).mean().item(),
        }
        self._reset_batch_state()
        return result
    
    def finalize_task(self, transition, global_info):
        print("\033[1;33mtransition reward: {}\033[0m".format(transition.get('reward', 0)))
        if self.execution_count <= 0 or not self.executed_trajectories:
            logger.warning("Ignoring finalize_task(): no active trajectory")
            return False

        self.current_trajectories = self.executed_trajectories[self.execution_count-1]
        idx = transition.get('path_id', 0)
        if not self.current_trajectories or idx >= len(self.current_trajectories):
            logger.warning("Ignoring finalize_task(): trajectory path does not exist")
            return False

        current_trajectory = self.current_trajectories[idx]
        if not current_trajectory:
            logger.warning("Ignoring finalize_task(): trajectory is empty")
            return False
        if current_trajectory[-1].get('finalized', False):
            logger.warning("Ignoring finalize_task(): trajectory is already finalized")
            return False

        termination_reason = transition.get('termination_reason')
        if termination_reason not in {"policy_stop", "max_steps"}:
            raise ValueError(f"Unknown termination reason: {termination_reason}")

        last_transition = current_trajectory[-1]
        terminator_role = self.agent_role_list[self.end_action.item()]
        if termination_reason == "policy_stop" and last_transition.get("action") != terminator_role:
            raise ValueError("policy_stop must correspond to a router-selected Terminator action")
        if termination_reason == "max_steps" and last_transition.get("action") == terminator_role:
            raise ValueError("max_steps cannot synthesize or finalize a Terminator action")

        workflow_actions = global_info.workflow.workflow
        if len(workflow_actions) != len(current_trajectory):
            raise ValueError(
                "Finalized trajectory must correspond one-to-one with executed workflow actions"
            )
        for trajectory_item, workflow_action in zip(current_trajectory, workflow_actions):
            executed_role = getattr(
                workflow_action, 'agent_role', getattr(workflow_action, 'role', None)
            )
            if trajectory_item.get('action') != executed_role:
                raise ValueError(
                    "Trajectory action does not match the executed workflow action"
                )

        routing_agent_calls = sum(
            getattr(action, 'agent_role', getattr(action, 'role', None)) != terminator_role
            for action in workflow_actions
        )
        aggregation_calls = int(transition.get('aggregation_calls', 0))
        routing_tokens = global_info.total_tokens
        aggregation_tokens = int(transition.get('aggregation_tokens', 0))
        total_tokens = routing_tokens + aggregation_tokens
        total_cost = global_info.total_cost
        task_reward = torch.tensor(
            float(transition.get('reward', self.task_reward_incorrect)),
            dtype=torch.float32,
            device=self.device,
        )

        if termination_reason == "policy_stop":
            # The sampled Terminator is a real policy action.  It has no call
            # cost and receives the task correctness reward exactly once.
            last_transition['reward'] = task_reward
        else:
            # max_steps is an environment event.  Keep the last real agent's
            # uniform call cost and expose task correctness as a separate
            # terminal reward with no action or log_prob.
            last_transition['terminal_reward'] = task_reward
        last_transition['task_reward'] = task_reward
        last_transition['routing_agent_calls'] = routing_agent_calls
        last_transition['aggregation_calls'] = aggregation_calls
        last_transition['total_llm_calls'] = routing_agent_calls + aggregation_calls
        last_transition['routing_tokens'] = routing_tokens
        last_transition['aggregation_tokens'] = aggregation_tokens
        last_transition['total_tokens'] = total_tokens
        last_transition['total_cost'] = total_cost
        last_transition['finalized'] = True
        last_transition['termination_reason'] = termination_reason
        last_transition['metrics'] = transition.get('metrics', {})
        print("\033[1;33mTask Reward: {}\033[0m".format(task_reward))
        self.rewards_history.append(float(task_reward.item()))
        return True
    
    def load_model(self, path, strict=True):
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            checkpoint = torch.load(path, map_location=self.device)
            
            # Validate model architecture before loading any policy parameters.
            if checkpoint['output_dim'] != self.policy_network.output_dim:
                raise ValueError(f"Policy output dimension mismatch. Expected output_dim={self.policy_network.output_dim} "
                                 f"for the sequential agent space, but got output_dim={checkpoint['output_dim']}")
            if checkpoint['input_dim'] != self.policy_network.input_dim:
                if strict:
                    raise ValueError(f"Model architecture mismatch. Expected input_dim={self.policy_network.input_dim}, "
                                     f"output_dim={self.policy_network.output_dim} but got input_dim={checkpoint['input_dim']}, "
                                     f"output_dim={checkpoint['output_dim']}")
                logger.warning("Model input dimension mismatch, but continuing due to non-strict mode")
            
            # Load model state
            # self.policy_network.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            self.policy_network.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.policy_network = self.policy_network.to(self.device)
            
            # Load optimizer state if available
            if checkpoint['optimizer_state_dict'] and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            
            # Load config if available
            if 'config' in checkpoint:
                # Merge loaded config with current config, prioritizing current config
                self.config.update({k: v for k, v in checkpoint['config'].items() 
                                  if k not in self.config})
            
            logger.info(f"Model loaded successfully from {path}")
            logger.info(f"Model timestamp: {checkpoint['timestamp']}")
            if checkpoint['metadata'].get('tag'):
                logger.info(f"Model tag: {checkpoint['metadata']['tag']}")
                
            return True
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def get_latest_model_path(self):
        """Get the path of the latest model checkpoint"""
        try:
            path = self.model_path
            if os.path.exists(path) and os.path.isfile(path):
                return path
            
            path = self.config["paths"]["checkpoint_path"]
            if not os.path.exists(path):
                return None

            model_files = [f for f in os.listdir(path) if f.endswith('.pt')]
            if not model_files:
                return None
            
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(path, x)))
            return os.path.join(path, latest_model)
            
        except Exception as e:
            print(f"Error finding latest model: {str(e)}")
            return None
    
