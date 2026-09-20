from typing import List
import json
import yaml
import os
import copy
import logging

from inference.reasoning.path import ReasoningState, GraphReasoningPath
from inference.graph.agent_graph import AgentGraph
from inference.graph.action_graph import ActionGraph
from inference.policy.REINFORCE_continuous import ContinuousREINFORCE

from utils.logging import LogManager

from agent.register.register import agent_global_registry
from agent.agent_info.global_info import GlobalInfo

from tasks.evaluator import BenchmarkEvaluator

global_config = yaml.safe_load(open("puppeteer/config/global.yaml", "r"))
main_logger = logging.getLogger('global') 

class GraphReasoning:
    def __init__(self, task:json, graph: AgentGraph, env=None, env_name=None):
        self.task = task
        self.agent_graph = graph
        self.action_graph = ActionGraph()
        self.reasoning_paths: List[GraphReasoningPath] = []
        
        # Compatibility value only; sequential reasoning always keeps one path.
        self.max_parallel_paths = global_config.get("graph").get("max_parallel_paths")
        
        self.final_answer = ""
        self.answers = []

        self.global_logger = LogManager("puppeteer/config/global.yaml", self.task.get("type"))

        self.workspace_path = self.global_logger.folder_path
        self.policy = ContinuousREINFORCE(agent_graph=self.agent_graph, action_graph=self.action_graph)

        self.env = env
        self.env_name = env_name
        main_logger.info("{}[Graph Reasoning Initialized]{}".format("-"*30, "-"*30))
        main_logger.info(global_config)
        main_logger.info(self.agent_graph.role_nodes)
    
    def save_checkpoint(self, save_data):
        main_logger.info("{}[Save Checkpoint]{}".format("-"*30, "-"*30))
        cur_acc = save_data["best_acc"]
        cur_data_len =  save_data["best_data_len"]
        main_logger.info("best acc: {}, data len: {}".format(cur_acc, cur_data_len))
        tag = "acc_{}-data_{}".format(cur_acc, cur_data_len)
        self.policy.save_model(path=None, tag=tag)

    def start(self, save_data):
        if save_data != None:
            self.save_checkpoint(save_data)
        print("-"*10+"\033[1;31mGraph Reasoning Start\033[0m"+"-"*10)
        main_logger.info("{}[Graph Reasoning Start]{}".format("-"*30, "-"*30))
        main_logger.info("Task:\n{}".format(self.task.get("Question")))
        
        global_info = GlobalInfo(path_id=-1, 
                                    workpath=self.workspace_path, 
                                    task=self.task, 
                                    env=self.env, 
                                    env_name=self.env_name)
        matches = self.policy.forward(global_info)
        assert len(matches) == 1, "Sequential reasoning must start with one agent"
        assert len(self.reasoning_paths) == 0, "GraphReasoning.start() can only initialize one path"

        index = 0
        match = matches[0]
        global_info = GlobalInfo(path_id=index,
                                workpath=self.workspace_path,
                                task=self.task,
                                env=self.env,
                                env_name=self.env_name)
        agent = agent_global_registry.get_agent_from_idx(match)
        assert agent is not None, "Selected start agent must exist in the registry"
        agent.activate(global_info)
        main_logger.info("[Path {} Initialized".format(index))
        print("\033[1;36mPath {} Initialized\033[0m".format(index))

        reasoning_path = GraphReasoningPath(start_agent=agent,
                                            max_parallel_paths=self.max_parallel_paths,
                                            action_graph=self.action_graph,
                                            agent_sequence=[],
                                            index=index,
                                            global_info=copy.deepcopy(global_info),
                                            global_logger=self.global_logger,
                                            workspace_path=self.workspace_path,
                                            state=copy.deepcopy(ReasoningState.INITIALIZED),
                                            env=self.env,
                                            env_name=self.env_name,
                                            policy=self.policy
                                            )
        self.reasoning_paths.append(reasoning_path)
        assert len(self.reasoning_paths) == 1, "Sequential reasoning must keep one path"
        main_logger.info("Reasoning Path: {}\nAgent Sequence: {}\n".format(index, reasoning_path.print_agent_sequence()))
    
    def n_step(self, n:int):
        for i in range(n):
            self.step()
            if self.check_finalize():
                break
        return self.finalize()
    
    def step(self):
        main_logger.info("{}[STEP]{}".format("-"*30, "-"*30))

        assert len(self.reasoning_paths) == 1, "Sequential reasoning must have exactly one path before each step"
        reasoning_path = self.reasoning_paths[0]
        assert reasoning_path.state != ReasoningState.SPLITING, "Sequential reasoning cannot enter SPLITING"

        if reasoning_path.state != ReasoningState.FINALIZING and reasoning_path.state != ReasoningState.DISCARDING:
            main_logger.info("{}[Reasoning Path{} STEP]{}".format("-"*30, reasoning_path.index, "-"*30))
            print("\033[1;36mPath {} Step\033[0m".format(reasoning_path.index))
            reasoning_path.step()
            main_logger.info("{}[DONE]: Reasoning Path{} STEP{}".format("-"*30, reasoning_path.index, "-"*30))

        assert len(self.reasoning_paths) == 1, "Sequential reasoning must have exactly one path after each step"
        assert reasoning_path.state != ReasoningState.SPLITING, "Sequential reasoning cannot enter SPLITING"
        if reasoning_path.state == ReasoningState.FINALIZING:
            print("\033[1;36mPath {} Finalize\033[0m".format(reasoning_path.index))
            main_logger.info("{}[Reasoning Path{} FINALIZING]{}".format("-"*30, reasoning_path.index, "-"*30))


        self.print_paths()
        self.update_graph()
        
        # return self.answers

    def aggregate_answers(self, global_info, answers:list, query_func=None) -> str:
        # only choose the last result without any format or extract
        if query_func is None:
            if len(answers) == 0:
                return None
            else:
                main_logger.info("[Aggregation] {}".format(answers[-1]))
                return answers[-1] 
        
        prompt_filepath = "puppeteer/prompts/general/answer_prompt.json" 
        with open(prompt_filepath, "r") as f:
            prompt = json.load(f)
        
        if self.task.get("type") == "MMLU" or self.task.get("type") == "MMLU-Pro":
            answer_prompt =  "\n".join(prompt["MMLU_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        elif self.task.get("type") == "GAIA":
            answer_prompt =  "\n".join(prompt["GAIA_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        elif self.task.get("type") == "GSM-Hard"  or self.task.get("type") == "gsm-hard" or self.task.get("type") == "GSM8K":
            answer_prompt = "\n".join(prompt["gsm_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        else: 
            answer_prompt = "\n".join(prompt["answer_aggregation"]).format(str(["{}\n".format(answer) for answer in answers]))
        
        main_logger.info("[Aggregating] {}".format(answer_prompt))
        
        raw_response, _ = query_func(messages=answer_prompt)
        main_logger.info("[Aggregation Answer] {}".format(raw_response))
        
        return raw_response if len(raw_response)!=0 else answers[-1]

    def finalize(self):
        print("-"*10+"\033[1;31mGraph Reasoning Finalize\033[0m"+"-"*10)
        assert len(self.reasoning_paths) == 1, "Sequential finalization must use exactly one path"
        reasoning_path = self.reasoning_paths[0]
        idx = 0
        should_update_policy = True

        if hasattr(reasoning_path, "last_query_func"):
            aggregated_answer = self.aggregate_answers(reasoning_path.global_info, reasoning_path.global_info.state_answers, reasoning_path.last_query_func)
        else:
            aggregated_answer = self.aggregate_answers(reasoning_path.global_info, reasoning_path.global_info.state_answers)

        if self.task.get("type") == "MMLU-Pro":
            transition = {
            'state': reasoning_path.global_info.workflow.state,
            'reward': 1 if BenchmarkEvaluator.check_mmlu(aggregated_answer, self.task.get("Answer")) else -1,
            'action': None,
            'next_state': None,
            'done': True,
            'path_id': idx,
            'termination_reason': reasoning_path.termination_reason,
            }
            print(transition)
            should_update_policy = self.policy.finalize_task(transition, reasoning_path.global_info)
        elif self.task.get("type") == "GSM-Hard":
            transition = {
            'state': reasoning_path.global_info.workflow.state,
            'reward': 1 if BenchmarkEvaluator.check_gsm8k(aggregated_answer, self.task.get("Answer")) else -1,
            'action': None,
            'next_state': None,
            'done': True,
            'path_id': idx,
            'termination_reason': reasoning_path.termination_reason,
            }
            print(transition)
            should_update_policy = self.policy.finalize_task(transition, reasoning_path.global_info)


        if aggregated_answer is not None:
            self.answers.append(aggregated_answer)
            main_logger.info("[Aggregated Answer From Path {}]: {}".format(idx, aggregated_answer))
        if should_update_policy:
            self.policy.update()
        
        for agent in agent_global_registry.agents.values():
            agent.reset()
        
        if len(self.answers) == 0:
            self.final_answer = ""
        else:
            self.final_answer = aggregated_answer
        
        main_logger.info("[Final Answer]: {}".format(self.final_answer))   
        print("-"*10+"\033[1;31mGraph Reasoning Finalized\033[0m"+"-"*10)
        
        return self.final_answer, self.task.get("Answer")
    
    def visualize_path(self):
        for reasoning_path in self.reasoning_paths:
            reasoning_path.global_info.workflow.visualize()
    
    def visualize_graph(self):
        self.agent_graph.visualize(os.path.join(self.workspace_path, "agent_graph.html"))
        self.action_graph.visualize(os.path.join(self.workspace_path, "action_graph.html"))

    def print_paths(self):
        for reasoning_path in self.reasoning_paths:
            main_logger.info("Reasoning Path: {}\nAgent Sequence: {}\n".format(reasoning_path.index, reasoning_path.print_agent_sequence()))
    
    def format_index(self):
        for index, reasoning_path in enumerate(self.reasoning_paths):
            reasoning_path.index = index
    
    def update_graph(self):
        for index, reasoning_path in enumerate(self.reasoning_paths):
            for successor, predecessor in zip(reasoning_path.agent_sequence[:-1], reasoning_path.agent_sequence[1:]):
                successor = agent_global_registry.get_agent_from_idx(successor.get("hash"))
                predecessor = agent_global_registry.get_agent_from_idx(predecessor.get("hash"))
                res = self.agent_graph._get_edge(predecessor, successor)
                if res is None or index not in res:
                    self.agent_graph._add_edge(predecessor, successor, index)
    
    def check_finalize(self):
        assert len(self.reasoning_paths) == 1, "Sequential reasoning must finalize exactly one path"
        reasoning_path = self.reasoning_paths[0]
        return reasoning_path.state == ReasoningState.FINALIZING or reasoning_path.state == ReasoningState.DISCARDING
