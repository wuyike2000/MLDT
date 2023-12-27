import copy
import glob
import os, sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import pdb
import pickle
import json
import random
from copy import deepcopy

from utils_bc import utils_interactive_eval
from utils_bc.utils_graph import filter_redundant_nodes
from envs.utils.check_logical import check_env_bug
from llm_policy import LLMPolicy, split_goal
from sim_compute import Similarity

def compute_task_complexity(task_goal, graph):
    min_steps = 0
    for goal in task_goal:
        goal_num = task_goal[goal]
        if 'turn' in goal:
            min_steps += 1
        elif 'inside' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            # indide object num
            inside_num = 0
            # outside object num
            out_num = 0
            # judge object location
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            # use object outside first, due to its fewer action step 
            # obj inside: walk, open, grab, close, walk, open, putin, close
            # obj outside: walk, grab, walk, open, putin, close
            if obj_num <= out_num:
                min_steps += 6 * goal_num
            else:
                min_steps += 6 * out_num + 8 * (obj_num - out_num)
        elif 'on' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            inside_num = 0
            out_num = 0
            # pan duan obj wei zhi
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            # use object outside first, due to its fewer action step 
            # obj inside: walk, open, grab, close, walk, putback
            # obj outside: walk, grab, walk, putback
            if obj_num <= out_num:
                min_steps += 4 * obj_num
            else:
                min_steps += 4 * out_num + 6 * (obj_num - out_num)
    return min_steps
    
def data_collection(args, vh_envs, logging):
    llm_policy = LLMPolicy(args,logging)
    # control flags
    if_exe_all_action = True
    verbose = True
    valid_run = 0
    success_count = 0
    data_collection = []
    camera_num = vh_envs.comm.camera_count()[1]
    
    # set test_examples
    origin=pickle.load(open('../data/test_init_env/'+args.subset+'.p','rb'))
    args.test_examples=len(origin)
    del origin
    
    # task number
    i = 0
    # iterate task
    while i < args.test_examples:
            
        print('*'*30,"New Sample","*"*30)
        
        # set retry
        retry=0
        while retry<args.max_retry:
            all_cur_observation = []
            all_actions = []
            all_rewards = []
            all_frames = []
    
            obs, env_graph = vh_envs.reset(task_id=i)
            obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
            all_cur_observation.append(deepcopy(obs))
    
            steps = 0
    
            valid_run_tem = False
            success_run_tem = False
    
            # -----------compute task complexity-------------------------
            task_goal = vh_envs.task_goal[0]
            graph = env_graph
            complexity = compute_task_complexity(task_goal, graph)
            print('Current Task id: {}, Task Goal: {}, Min Step: {}'.format(i,task_goal,complexity))
            # --------------
            # set gpt policy
            # --------------
            llm_policy.reset(ids=i)
            llm_policy.set_graph(env_graph)  # 设置gpt任务环境
            llm_policy.set_goal(vh_envs.task_goal[0])  # 设置gpt任务目标
            
            if if_exe_all_action:
                llm_policy.generate_multi_layer_plan()  # 生成gpt任务规划
            else:
                llm_policy.split_task_goal, llm_policy.split_task_goal_num = split_goal(logging,
                                                                                        llm_policy.task_goal)
            
            print('*'*20,'Execute Action List','*'*20)
            # iterate generated action
            while True:
                agent_id = 0
                agent_actions = {}
                agent_rewards = {}
                agent_ignore_walk = {}
                ignore_walk = None
                action_obj_str = ''
                if if_exe_all_action:
                    # collect all the action generated by gpt
                    llm_action_obj_str = llm_policy.get_action_from_llm()
                    if llm_action_obj_str != 'DONE':
                        env_task_goal_write = ['%s_%d' % (k, v) for k, v in vh_envs.task_goal[0].items() if v > 0]
                        print('-'*100)
                        print('Step: {}, Task: {}'.format(steps+1,str(env_task_goal_write)))
                        print('Action: {}'.format(llm_action_obj_str))
                    else:
                        valid_run_tem = True
                        break
                else:
                    llm_action_obj_str = llm_policy.get_action_from_llm()
                    if llm_action_obj_str == '':
                        # generate plan of the subtask
                        if llm_policy.goal_exe_index < llm_policy.split_task_goal_num:
                            current_task = llm_policy.split_task_goal[llm_policy.goal_exe_index]
                            llm_policy.goal_exe_index += 1
                            llm_policy.generate_plan(current_task)
                        llm_action_obj_str = llm_policy.get_action_from_llm()
                action_obj_str = llm_action_obj_str
                agent_actions[agent_id] = action_obj_str
                agent_ignore_walk[agent_id] = ignore_walk
    
                ## ----------------------------------------------------------------------------------------------------
                ## send action to the environment
                ## ----------------------------------------------------------------------------------------------------
                obs, rewards, dones, infos, success = vh_envs.step(agent_actions, ignore_walk=agent_ignore_walk,
                                                                   logging=logging)  # next_obs
    
                if rewards == dones == infos == success == None:
                    print('Action Fail')
                    print('Fail Reason: {}'.format(json.dumps(obs)))
                    valid_run_tem = False
                    break
    
                ## ---------------------------------------------------------------------------------------------------------
                ## check action after send to Unity
                ## ---------------------------------------------------------------------------------------------------------
                obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
                env_bug_count_a0 = not check_env_bug(agent_actions[0], obs[0], agent_i=0, logging=logging)
    
                if env_bug_count_a0:
                    print('Action Fail')
                    print('check_env_bug outside unity fail!')
                    valid_run_tem = False
                    break
    
                ## ----------------------------------------------------------------------------------------------------
                ## done, bad end
                ## ----------------------------------------------------------------------------------------------------
                all_cur_observation.append(deepcopy(obs))
                all_actions.append(deepcopy(agent_actions))
                print('Action Success')
    
                ## ---------------------------------------------------------------------------------------------------------
                ## log
                ## ---------------------------------------------------------------------------------------------------------
                # if verbose:
                #    env_task_goal_write = ['%s_%d' % (k, v) for k, v in vh_envs.task_goal[0].items() if v > 0]
                    # logging.info('task %d, step %d, goal %s' % (i, steps, str(env_task_goal_write)))
                    # logging.info(('Act: %s' % str(agent_actions[0])))
    
                ## ---------------------------------------------------------------------------------------------------------
                ## break if done
                ## ---------------------------------------------------------------------------------------------------------
                steps += 1
                if np.any(dones):
                    valid_run_tem = True
    
                    if infos[0]['is_success']:
                        success_run_tem = True
                    break

            # judge whether need to retry
            # success
            if success_run_tem:
                # record the data
                if args.collection:
                    temp=dict()
                    temp['task_id']=llm_policy.task_id
                    temp['task_goal']=llm_policy.task_goal
                    temp['split_task_goal']=llm_policy.split_task_goal
                    temp['subtask']=llm_policy.subtask
                    temp['action_lists']=llm_policy.exec_action_lists
                    temp['partial_locate']=llm_policy.partial_locate
                    temp['partial_state']=llm_policy.partial_state
                    temp['task_prompt']=llm_policy.task_prompt
                    temp['task_res']=llm_policy.task_res
                    temp['action_prompt']=llm_policy.action_prompt
                    temp['action_res']=llm_policy.action_res                  
                    data_collection.append(temp)
                break
            else:
                print('*'*20,'Plan Fail','*'*20)
                retry+=1
                # if already exceed max retry
                if retry>=args.max_retry:
                    print('Reach Max Retry')
                else:
                    valid_run_tem = False
                    success_run_tem = False
                    print('Retry: {} Max Retry: {}'.format(retry,args.max_retry))
            
        # plan result
        print('*'*20,'Plan Results','*'*20)
        if valid_run_tem:
            valid_run += 1
            print('Executable Plan')

            if success_run_tem:
                success_count += 1
                print('Successful Plan')
            else:
                print('Plan is not successful')
                
        else:
            print('Plan is not executable')
        
            
        # increase task number
        i+=1

        if args.interactive_eval:
            execute_rate = 100. * valid_run / i if i != 0 else 0
            success_rate = 100. * success_count / i if i != 0 else 0
            print('*'*10,'Current Evaluation Metric','*'*10)
            print('Successful / Executable / Current / Total: {} / {} / {} / {}'.format(success_count,valid_run,i,args.test_examples))
            print('Success Rate: {}'.format(success_rate))
            print('Executability: {}'.format(execute_rate))
        
        # force write to log
        sys.stdout.flush()
        
        # save the data intervally to avoid no result when the program breaks up
        if args.collection and i%args.interval==0:
            json.dump(data_collection,open(args.output[:-5]+'-'+str(i)+'.json','w',encoding='utf-8'),indent=2)

    # save the data
    if args.collection:
        json.dump(data_collection,open(args.output,'w',encoding='utf-8'),indent=2)
    return success_rate