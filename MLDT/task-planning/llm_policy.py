import random
import time
import requests
import json
import re
import sys
import openai
import os
import torch
from sim_compute import Similarity

sys.path.append('../virtualhome')
from simulation.unity_simulator import comm_unity as comm_unity
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModel
from peft import PeftModel

# generation config
generation_config = GenerationConfig(
        temperature=0.001,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        #max_new_tokens=32768
        #max_new_tokens=4096
        max_length=32768    
)

# split goal according to id
def split_goal(log, task_goal):
    print('*'*20,'Current Step: Goal Decomposition','*'*20)
    id_dict=dict()
    items = task_goal.split(',')[:-1]
    for item in items:
        ids=re.findall('\(id:(\d+)\)',item)[0]
        if ids in id_dict:
            id_dict[ids].append(item)
        else:
            id_dict[ids] = [item]
    result_list = [','.join(group) for group in id_dict.values()]
    print('*'*10,'Original Task Goal','*'*10)
    print(task_goal)
    print('*'*10,'Sub Task Goal','*'*10)
    print(result_list)
    return result_list, len(result_list)

# def a class as a gpt policy
class LLMPolicy:
    def __init__(self, args, logging):
        # load llama-2-7b/bloom
        if 'llama-2-7b' in args.llm or 'bloom' in args.llm or 'LongAlpaca-7B' in args.llm:
            # load tokenizer and llm
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm, legacy=True)
            if not args.lora:
                self.llm=AutoModelForCausalLM.from_pretrained(args.llm,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',max_memory={0: "30GB"})
            else:
                base_model=AutoModelForCausalLM.from_pretrained(args.llm,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',max_memory={0: "30GB"})
                self.llm=PeftModel.from_pretrained(base_model, args.lora,torch_dtype=torch.float16,device_map='auto',max_memory={0: "30GB"})
        # load llama-2-13b
        if 'llama-2-13b' in args.llm or 'LongAlpaca-13B' in args.llm:
            # load tokenizer and llm
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm, legacy=True)
            if not args.lora:
                self.llm=AutoModelForCausalLM.from_pretrained(args.llm,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',max_memory={0: "30GB",1: "30GB"})
            else:
                base_model=AutoModelForCausalLM.from_pretrained(args.llm,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',max_memory={0: "30GB",1: "30GB"})
                self.llm=PeftModel.from_pretrained(base_model, args.lora,torch_dtype=torch.float16,device_map='auto',max_memory={0: "30GB",1: "30GB"})
        # load chatglm
        if 'chatglm' in args.llm:
            # load tokenizer and llm
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
            if not args.lora:
                self.llm=AutoModel.from_pretrained(args.llm,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',max_memory={0: "30GB"},trust_remote_code=True)
            else:
                base_model=AutoModel.from_pretrained(args.llm,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map='auto',max_memory={0: "30GB"},trust_remote_code=True)
                self.llm=PeftModel.from_pretrained(base_model, args.lora)
        # load closed source llm
        if 'gpt' in args.llm:
            self.tokenizer=None
            self.llm=args.llm
        self.graph = None
        self.task_id=None
        self.task_goal = None
        self.split_task_goal = None
        self.split_task_goal_num = 0
        self.subtask=[]
        self.goal_exe_index = 0
        self.task_obj = []  #   ¼          ?   obj
        self.exec_action_lists = []
        self.exec_action_index = 0  # the index of the action to be executed
        self.goal_objs_loc = None
        self.logging = logging
        self.api=args.api
        self.mode=args.mode

        # use demo or not
        self.demo=args.demo
        if args.demo:
        # similarity module
            self.sc = Similarity()
            # action demo
            self.action_demo=[]
            self.action_match=[]
            data=json.load(open('demo/action.json','r',encoding='utf-8'))
            for i in data:
                self.action_demo.append(i["content"])
                self.action_match.append(i["match"])
            # task demo
            self.task_demo=[]
            self.task_match=[]
            data=json.load(open('demo/task.json','r',encoding='utf-8'))
            for i in data:
                self.task_demo.append(i["content"])
                self.task_match.append(i["match"])
            # react demo
            self.react_demo=[]
            self.react_match=[]
            data=json.load(open('demo/react.json','r',encoding='utf-8'))
            for i in data:
                self.react_demo.append(i["content"])
                self.react_match.append(i["match"])
            # embodied demo
            self.embodied_demo=[]
            self.embodied_match=[]
            data=json.load(open('demo/embodied.json','r',encoding='utf-8'))
            for i in data:
                self.embodied_demo.append(i["content"])
                self.embodied_match.append(i["match"])
            # goal-action demo
            self.goal_action_demo=[]
            self.goal_action_match=[]
            data=json.load(open('demo/goal-action.json','r',encoding='utf-8'))
            for i in data:
                self.goal_action_demo.append(i["content"])
                self.goal_action_match.append(i["match"])   
            # task-action demo
            self.task_action_demo=[]
            self.task_action_match=[]
            data=json.load(open('demo/task-action.json','r',encoding='utf-8'))
            for i in data:
                self.task_action_demo.append(i["content"])
                self.task_action_match.append(i["match"])                                                         
            '''
            self.demo_task=[]
            self.demo_task_goal=[]
            with open('subtask_demo/v2.txt','r',encoding='utf-8') as f:
                text=f.read()
            for demo in text.split('--------------------------------------------------------------------------'):
                self.demo_task.append(demo.strip())
            for demo in self.demo_task:
                for line in demo.split('\n'):
                    if line.startswith('# task goal: '):
                        self.demo_task_goal.append(line.strip().replace('# task goal: ',''))
                        break
            self.demo_plan=[]
            self.demo_plan_goal=[]
            with open('react_demo/react_demo_v4.txt','r',encoding='utf-8') as f:
                text=f.read()
            for demo in text.split('--------------------------------------------------------------------------'):
                self.demo_plan.append(demo.strip())
            for demo in self.demo_plan:
                for line in demo.split('\n'):
                    if line.startswith('# task:'):
                        goal=line.strip().replace('# task:','')
                        # mask the object
                        obj = re.findall(r'(grab|put) (.+) (in|on)', goal)
                        if obj:
                            mask_goal=goal.replace(obj[0][1],'something')
                            mask_goal=re.sub('\(id:\d+\)','',mask_goal)
                            self.demo_plan_goal.append(mask_goal)
                        else:
                            self.demo_plan_goal.append(re.sub('\(id:\d+\)','',goal))
                        break  
            '''
        '''
        # rule construction
        self.rule=dict()
        rulelist=os.listdir('rule')
        for i in rulelist:
            key=i.replace('.txt','')
            self.rule[key]=[]
            with open('rule/'+i,'r',encoding='utf-8') as f:
                for line in f.readlines():
                    self.rule[key].append(line.strip())
        '''
        # data collection
        # input and output of task decomposition
        self.task_prompt=[]
        # without demo
        self.task_res=[]
        # input and output of action decomposition
        self.action_prompt=[]
        # without demo
        self.action_res=[]
        # partial observation
        self.partial_locate=[]
        self.partial_state=[]
        
    def reset(self,ids):
        self.graph = None
        self.task_id=ids
        self.task_goal = None
        self.split_task_goal = None
        self.split_task_goal_num = 0
        self.subtask=[]
        self.goal_exe_index = 0
        self.task_obj = []  #   ¼          ?   obj
        self.exec_action_lists = []
        self.exec_action_index = 0  # the index of the action to be executed
        self.goal_objs_loc = None
    
        # data collection
        # input and output of task decomposition
        self.task_prompt=[]
        # without demo
        self.task_res=[]
        # input and output of action decomposition
        self.action_prompt=[]
        # without demo
        self.action_res=[]
        # partial observation
        self.partial_locate=[]
        self.partial_state=[]
        
    def getLLMResponse(self,prompt,max_retries=1000):
        # open source llm
        if self.tokenizer:
            inputs = self.tokenizer(prompt,return_tensors="pt")
            generation_output = self.llm.generate(
                    input_ids=inputs["input_ids"].to('cuda'),
                    attention_mask=inputs['attention_mask'].to('cuda'),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    generation_config=generation_config
                )
            output = self.tokenizer.decode(generation_output[0],skip_special_tokens=True)
            response = output.split(prompt)[-1].strip()
            return response
        
        # closed source llm
        else:
            # set openai key
            openai.api_key = self.api
            # adopt external interface
            # openai.api_base = "https://api.aigcbest.top/v1"
            
            # set retries
            retries=0
            while retries < max_retries:
                try:
                    res = openai.ChatCompletion.create(
                        model=self.llm,
                        messages=[
                            {'role': 'user', 'content': prompt}
                        ],
                        temperature=0,
                    )
                    return res['choices'][0]['message']['content']
                except Exception as e:
                    print(f"Exception caught: {e}")
                    retries += 1
                    time.sleep(5)
    
    def set_graph(self, graph):
        self.graph = graph

    def set_goal(self, lid_goals):
        task_goal = ''
        goal_objs = []  # Ŀ    Ʒ
        # translate the env_task_goal_write to natural language
        for k, v in lid_goals.items():
            if v > 0:
                obj_id = int(k.split('_')[-1])
                obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == obj_id][0]
                #  жϵ ǰobj Ƿ   goal   Ѵ  ڱ    ظ   ?
                have_exist_in_goal = False
                for id, name in goal_objs:
                    if id == obj_id:
                        have_exist_in_goal = True
                if not have_exist_in_goal:
                    goal_objs.append((obj_id, obj_name))
                #  жϵ ǰgoal_obj  task_obj   Ƿ  Ѵ ?
                have_exist = False
                for id, name in self.task_obj:
                    if id == obj_id and obj_name == name:
                        have_exist = True
                if not have_exist:
                    self.task_obj.append((obj_id, obj_name))
                task_goal += k.replace(k.split('_')[-1], obj_name + "(id:{})".format(obj_id)) + ': ' + str(v) + ','
                #   ȡ   obj
                name = str(k.split('_')[-2])
                for node in self.graph['nodes']:
                    if node['class_name'] == name:
                        goal_objs.append((node['id'], name))

        # print('[INFO] task goal GPT version:', task_goal)
        self.task_goal = task_goal
        # print('[INFO] goal obj:')
        # find the location of the goal objects
        goal_objs_loc = []  #    еص  Ŀ    Ʒ
        for obj_id, obj_name in goal_objs:
            from_obj_edges = [edge for edge in self.graph['edges'] if edge['from_id'] == obj_id]
            for edge in from_obj_edges:
                if edge['relation_type'] == 'INSIDE':
                    to_obj_id = edge['to_id']
                    to_obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id][0]
                    self.task_obj.append((to_obj_id, to_obj_name))
                    goal_objs_loc.append(('%s(id:%d)' % (obj_name, obj_id), edge['relation_type'],
                                          '%s(id:%d)' % (to_obj_name, to_obj_id)))

        self.goal_objs_loc = goal_objs_loc
        self.task_goal = task_goal

    def get_goal_obj_message(self, task):
        # closed_microwave(id:158): 1,turnon_microwave(id:158): 1,on_milk_kitchentable(id:123): 3,inside_pancake_microwave(id:158): 1,
        goals = task.split(',')
        need_grab_obj = []
        # list of object location
        goal_objs_loc = []
        # list of object state
        goal_objs_state = []
        need_put_obj = []
        need_get_obj = []
        reason_message = []
        for goal in goals:
            obj = goal.split('_')
            for name in obj:
                for node in self.graph['nodes']:
                    if node['class_name'] == name:
                        need_grab_obj.append((node['id'], name))
                pattern = r'(\w+)\(id:(\d+)\)'
                matches = re.findall(pattern, name)
                if matches:
                    id_ = int(matches[0][1])
                    name_ = matches[0][0]
                    id_list = [id_ for id_, name_ in need_put_obj]
                    if id_ not in id_list:
                        need_put_obj.append((id_, name_))

        for obj_id, obj_name in need_grab_obj:
            reason_message.append('%s(id:%d)' % (obj_name, obj_id))
            from_obj_edges = [edge for edge in self.graph['edges'] if edge['from_id'] == obj_id]
            for edge in from_obj_edges:
                #print(edge)
                #print(obj_id)
                #print(obj_name)
                #print(edge['to_id'])
                #print([node['class_name'] for node in self.graph['nodes'] if node['id'] == edge['to_id']])
                if edge['relation_type'] == 'INSIDE':
                    #print(edge)
                    to_obj_id = edge['to_id']
                    #print([node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id])
                    to_obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id][0]
                    goal_objs_loc.append(('%s(id:%d)' % (obj_name, obj_id), edge['relation_type'],
                                          '%s(id:%d)' % (to_obj_name, to_obj_id)))
                    id_list = [id_ for id_, name_ in need_get_obj]
                    if to_obj_id not in id_list:
                        need_get_obj.append((to_obj_id, to_obj_name))
                
                
        # get relevant object state
        obj_state = ''
        for obj_id, obj_name in need_put_obj:
            state = ''
            reason_message.append('%s(id:%d)' % (obj_name, obj_id))
            for node in self.graph['nodes']:
                if node['id'] == obj_id:
                    state = node['states']
                    break
            if state != '':
                if 'OPENED' in state and 'ON' in state:
                    obj_state += '{}(id:{}) is open|on, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open|on'])
                    continue
                if 'OPENED' in state and 'OFF' in state:
                    obj_state += '{}(id:{}) is open|off, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open|off'])
                    continue
                if 'CLOSED' in state and 'ON' in state:
                    obj_state += '{}(id:{}) is closed|on, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed|on'])
                    continue
                if 'CLOSED' in state and 'OFF' in state:
                    obj_state += '{}(id:{}) is closed|off, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed|off'])
                    continue
                if 'OPENED' in state:
                    obj_state += '{}(id:{}) is open, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open'])
                    continue
                if 'CLOSED' in state:
                    obj_state += '{}(id:{}) is closed, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed'])
                    continue
                if 'ON' in state:
                    obj_state += '{}(id:{}) is on, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','on'])
                    continue
                if 'OFF' in state:
                    obj_state += '{}(id:{}) is off, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','off'])
                    continue
        
        for obj_id, obj_name in need_get_obj:
            state = ''
            for node in self.graph['nodes']:
                if node['id'] == obj_id:
                    state = node['states']
                    break
            if state != '':
                if 'OPENED' in state:
                    obj_state += '{}(id:{}) is open, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open'])
                    continue
                if 'CLOSED' in state:
                    obj_state += '{}(id:{}) is closed, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed'])
                    continue

        obj_loc=''
        for i in goal_objs_loc:
            obj_loc=obj_loc+i[0]+' is in '+i[2]+', '
            
        # record goal_objs_location and goal_bojs_state
        self.partial_locate.append(goal_objs_loc)
        self.partial_state.append(goal_objs_state)
        
        # return all relevant obj
        obj=[]
        obj.extend(need_grab_obj)
        obj.extend(need_put_obj)
        obj.extend(need_get_obj)
        return obj_loc[:-2], obj_state[:-2], obj

        # return str(reason_message)

    def get_subtask_message(self, reason_subtask):
        pattern = r"id:(\d+)"
        ids = re.findall(pattern, reason_subtask)
        goal_objs_loc = []
        need_get_obj = []
        obj_state = ''
        for id_ in ids:
            id_ = int(id_)
            from_obj_edges = [edge for edge in self.graph['edges'] if edge['from_id'] == id_]
            to_obj_edges = [edge for edge in self.graph['edges'] if edge['to_id'] == id_]
            nodes = [node for node in self.graph['nodes'] if node['id'] == id_]
            if nodes:
                obj_name = nodes[0]['class_name']
            else:
                return False
            for edge in from_obj_edges:
                if edge['relation_type'] == 'INSIDE':
                    to_obj_id = edge['to_id']
                    to_obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id][0]
                    goal_objs_loc.append(('%s(id:%d)' % (obj_name, id_), edge['relation_type'],
                                          '%s(id:%d)' % (to_obj_name, to_obj_id)))
                    ids.append(to_obj_id)
                    print(ids)
            state = ''
            for edge in to_obj_edges:
                if edge['relation_type'] == 'HOLDS_RH':
                    obj_state += '{}(id:{}) in your hand,'.format(obj_name, id_)
            for node in self.graph['nodes']:
                if node['id'] == id_:
                    state = node['states']
                    break
            if state != '':
                if 'OPENED' in state:
                    obj_state += '{}(id:{})\'s states are '.format(obj_name, id_)
                    obj_state += 'opened,'
                if 'CLOSED' in state:
                    obj_state += '{}(id:{})\'s states are '.format(obj_name, id_)
                    obj_state += 'closed,'
                if 'ON' in state:
                    obj_state += 'on,'
                if 'OFF' in state:
                    obj_state += 'off,'
        if obj_state:
            state_memory = str(list(set(goal_objs_loc))) + ' and ' + obj_state
        else:
            state_memory = str(list(set(goal_objs_loc)))
        return state_memory

    # get top k demonstration
    def get_demo(self, K, match, demo_list, match_list):
        # -----------------demo build---------------------------
        exampleTask = ''

        # Extract the string information of each task
        scores = []
        sim_score = self.sc.sim_compute(match,match_list)
        for index,demo in enumerate(demo_list):
            scores.append([sim_score[index], match_list[index], demo])

        scores.sort(reverse=True)

        topk = scores[:K]
        examplelist = [score[2] for score in topk]

        for demo in examplelist:
            exampleTask = exampleTask+demo.strip()+'\n\n'

        return exampleTask

    def generate_plan(self, task):
        print('*'*20,'Current Step: Generate plan for {}'.format(task),'*'*20)
        obs_loc,obs_state,obj=self.get_goal_obj_message(task)
        '''
        # global rule
        rulePrompt ='# rule 1: Different id represent different items, so note the id number\n# rule 2: Grab only one item at a time\n'
        index=3
        for i in obj:
            if self.rule.get(i[1]):
                for j in self.rule[i[1]]:
                    rulePrompt=rulePrompt+'# rule '+str(index)+': '+j+'\n'
                    index+=1
        '''
        actionPrimitives = "from actions import " \
                           "walk <obj>, grab <obj>, switchon <obj>, " \
                           "open <obj>, close <obj>, " \
                           "putin <obj> <obj>, putback <obj> <obj>\n"
        if self.demo:
            mask_task=re.sub('\(id:\d+\)','',task) 
            if self.mode=='react': 
                exampleTask = self.get_demo(3, mask_task,self.react_demo,self.react_match)
            if self.mode=='goal-action':
                exampleTask = self.get_demo(3, mask_task,self.goal_action_demo,self.goal_action_match)
        else:
            exampleTask=''
        # version 3: without rule
        #full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        # version 2: with global action and rule
        full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'
        # version 3: each example with action and rule
        #full_prompt=exampleTask+actionPrimitives+rulePrompt+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        next_prompt = "# task goal: " + task + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        # process context for it may have more than one sample
        if '# key object location:' in context:
            context=context.split('# key object location:')[0]
        print('*'*10,'Prompt','*'*10)
        print(final_prompt)
        print('*'*10,'LLM Output','*'*10)
        print(context)
        self.context_analysis(context)
        '''
        # modify the action list based on several rules
        self.rule_modify()
        # if task goal include turn on, make sure the last action is switchon
        if len(self.exec_action_lists)>0 and 'turnon' in next_prompt:
            # the last action is not switchon
            if 'switchon' not in self.exec_action_lists[-1]:
                # make sure we can find the switchon object
                if 'close' in self.exec_action_lists[-1]:
                    gd=self.exec_action_lists[-1].replace('close','switchon')
                    self.exec_action_lists.append(gd)
        '''
        
    # react
    def generate_react_plan(self):
        # generate plan for goal
        self.generate_plan(self.task_goal)
        print('*'*20,'Action List','*'*20)
        for action in self.exec_action_lists:
            print(action)

    # embodied
    def generate_embodied_plan(self):
        # generate plan for goal
        self.generate_embodied(self.task_goal)
        print('*'*20,'Action List','*'*20)
        for action in self.exec_action_lists:
            print(action)
            
    # goal-level decomposition + react
    def generate_goal_plan(self):
        # goal-level decomposition
        self.split_task_goal, self.split_task_goal_num = split_goal(self.logging, self.task_goal)
        # generate plan for subgoal
        for task in self.split_task_goal:
            self.generate_plan(task)
        print('*'*20,'Action List','*'*20)
        for action in self.exec_action_lists:
            print(action)
            
    # task-level decomposition + action-level decomposition
    def generate_task_plan(self):
        # task-level decomposition
        self.split_task(self.task_goal)
        # action-level decomposition
        for task in self.subtask:
            self.generate_subplan(task)        
        print('*'*20,'Action List','*'*20)
        for action in self.exec_action_lists:
            print(action)
    
    # MLDT
    def generate_multi_layer_plan(self):
        # goal-level decomposition
        self.split_task_goal, self.split_task_goal_num = split_goal(self.logging, self.task_goal)
        # task-level decomposition
        for goal in self.split_task_goal:
            self.split_task(goal)
        # action-level decomposition
        for task in self.subtask:
            self.generate_subplan(task)
        print('*'*20,'Action List','*'*20)
        for action in self.exec_action_lists:
            print(action)
    
    # task level decomposition
    def split_task(self,goal):
        print('*'*20,'Current Step: Task Decomposition for {}'.format(goal),'*'*20)                 
        # action
        actionPrimitives = "from action import grab <obj> in <obj>, put <obj> in <obj>, put <obj> on <obj>, switch on <obj>\n"
        
        exampleTask=''
        # demo
        if self.demo:
            mask_goal=re.sub('\(id:\d+\)','',goal)  
            if self.mode=='multi-layer':
                exampleTask = self.get_demo(3,mask_goal,self.task_demo,self.task_match)
            if self.mode=='task-action':
                exampleTask = self.get_demo(3,mask_goal,self.task_action_demo,self.task_action_match)
        
        # partial observation
        obs_loc,obs_state,obj=self.get_goal_obj_message(goal)
        # prompt construct
        full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'
        next_prompt = "# task goal: " + goal + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        # process context for it may have more than one sample
        if '# key object location:' in context:
            context=context.split('# key object location:')[0]
        print('*'*10,'Prompt','*'*10)
        print(final_prompt)
        print('*'*10,'LLM Output','*'*10)
        print(context)
        # analyze the context
        # if '```' in context:
        #     context=context.split('```')[0]
        for line in context.split('\n'):
            line=line.strip()
            
            if len(line)!=0 and not line.startswith('#') and line.split(' ')[0] in ['grab','put','switch']:
                match1=re.findall('(grab|put) (.+) (in|on) (.+)',line)
                match2=re.findall('switch on (.+)',line)
                if match1 or match2:
                    self.subtask.append(line)
            
            # if len(line)!=0 and not line.startswith('#'):
            #    self.subtask.append(line)
        # record 
        self.task_prompt.append([final_prompt.strip(),context.strip()])
        self.task_res.append([actionPrimitives+'\n'+'# key object location: '+obs_loc+'\n'+next_prompt.strip(),context.strip()])
    
    # action level decomposition: generate plan for subtask 
    def generate_subplan(self,task):
        print('*'*20,'Current Step: Generate plan for {}'.format(task),'*'*20)
        actionPrimitives = "from actions import " \
                           "walk <obj>, grab <obj>, switchon <obj>, " \
                           "open <obj>, close <obj>, " \
                           "putin <obj> <obj>, putback <obj> <obj>\n"
        
        exampleTask=''
        # demo
        if self.demo:
            # mask obj task
            obj = re.findall(r'(grab|put) (.+) (in|on)', task)
            if obj:
                mask_task=task.replace(obj[0][1],'something')
            else:
                mask_task=task
            mask_task=re.sub('\(id:\d+\)','',mask_task)
            if self.mode=='multi-layer' or self.mode=='task-action':
                exampleTask = self.get_demo(3,mask_task,self.action_demo,self.action_match)

        # version 3: without rule
        #full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        # version 2: with global action and rule
        full_prompt=actionPrimitives+'\n'+exampleTask
        # version 3: each example with action and rule
        #full_prompt=exampleTask+actionPrimitives+rulePrompt+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        next_prompt = "# task: " + task + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        # process context for it may have more than one sample
        if '# task:' in context:
            context=context.split('# task:')[0]
        print('*'*10,'Prompt','*'*10)
        print(final_prompt)
        print('*'*10,'LLM Output','*'*10)
        print(context)
        self.context_analysis(context)
        # record 
        self.action_prompt.append([final_prompt.strip(),context.strip()])
        self.action_res.append([actionPrimitives+'\n'+next_prompt.strip(),context.strip()])        

    def context_analysis(self, context):
        lines = context.split('\n')
        id_list = [] 
        for line in lines:
            line=line.replace(" ", "")
            pattern = r"(walk|find|open|grab|close|switchon)\('(\w+)\(id:(\d+)\)'\)"
            match = re.match(pattern, line)
            if match:
                action = match.group(1)
                if action == 'find':
                    action = 'walk'
                item_name = match.group(2)
                item_id = match.group(3)
                action_script = "[{}] <{}> ({})".format(action, item_name, item_id)
                self.exec_action_lists.append(action_script)
            pattern = r"(putback|putin)\('(\w+)\(id:(\d+)\)','(\w+)\(id:(\d+)\)'\)"
            match = re.match(pattern, line)
            if match:
                action = match.group(1)
                item1_name = match.group(2)
                item1_id = match.group(3)
                item2_name = match.group(4)
                item2_id = match.group(5)
                action_script = "[{}] <{}> ({}) <{}> ({})".format(action, item1_name, item1_id, item2_name, item2_id)
                self.exec_action_lists.append(action_script)
    
    # generate plan for embodied prompt
    def generate_embodied(self, task):
        print('*'*20,'Current Step: Generate plan for {}'.format(task),'*'*20)
        obs_loc,obs_state,obj=self.get_goal_obj_message(task)

        actionPrimitives = "from actions import " \
                           "walk <obj>, grab <obj>, switchon <obj>, " \
                           "open <obj>, close <obj>, " \
                           "putin <obj> <obj>, putback <obj> <obj>\n"
        if self.demo:
            mask_task=re.sub('\(id:\d+\)','',task)
            if self.mode=='embodied':
                exampleTask = self.get_demo(3, mask_task,self.embodied_demo,self.embodied_match)
        else:
            exampleTask=''
        # version 3: without rule
        #full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        # version 2: with global action and rule
        full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'
        # version 3: each example with action and rule
        #full_prompt=exampleTask+actionPrimitives+rulePrompt+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        next_prompt = "# task goal: " + task + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        # process context for it may have more than one sample
        if '# key object location:' in context:
            context=context.split('# key object location:')[0]
        print('*'*10,'Prompt','*'*10)
        print(final_prompt)
        print('*'*10,'LLM Output','*'*10)
        print(context)
        self.context_analysis(context)
    
    def rule_modify(self,max_loop=100):
        # judge satisfy rule or not
        FLAG=False
        
        # open/close object
        oc_obj=['cabinet','dishwasher','fridge','kitchencabinet','microwave','stove']
        
        # putin object
        putin_obj=oc_obj
        
        # putback object
        putback_obj=['sink','kitchentable']
        
        # switchon object
        switchon_obj=['stove','microwave','dishwasher']
        
        # set loop to avoid eternally loop
        loop=0
        
        # modify until not error occur
        while not FLAG and loop<max_loop: 
            FLAG=True
            loop+=1
            for index,i in enumerate(self.exec_action_lists):
                # match putin/putback
                matches = re.findall(r"\[(\w+)\] <(\w+)> \((\d+)\) <(\w+)> \((\d+)\)", i)
                if matches:
                    action=matches[0][0]
                    obj1=matches[0][1]
                    num1=matches[0][2]
                    obj2=matches[0][3]
                    num2=matches[0][4]
                else:
                    # match other actions
                    matches = re.findall(r"\[(\w+)\] <(\w+)> \((\d+)\)", i)
                    if matches:
                        action=matches[0][0]
                        obj=matches[0][1]
                        num=matches[0][2]
                
                # action after walk oc_obj must be open
                if action=='walk' and obj in oc_obj:
                    # gd
                    gd=i.replace('walk','open')
                    # not satisfy
                    if len(self.exec_action_lists)<=index+1:
                        self.exec_action_lists.append(gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index+1]!=gd:
                        self.exec_action_lists.insert(index+1,gd)
                        FLAG=False
                        break
                
                # action before grab must be walk, action after grab oc_obj must be close
                if action=='grab':
                    # gd before
                    gd=i.replace('grab','walk')
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
                    
                    # if interact object is in oc_obj
                    # extract object from previous step
                    if index>=2:
                        pre_step=self.exec_action_lists[index-2]
                        matches = re.findall(r"\[(\w+)\] <(\w+)> \((\d+)\)", pre_step)
                        if matches:
                            pre_act=matches[0][0]
                            pre_obj=matches[0][1]
                            pre_num=matches[0][2]
                            if pre_act=='open' and pre_obj in oc_obj:
                                # gd after
                                gd=pre_step.replace('open','close')
                                # not satisfy
                                if len(self.exec_action_lists)<=index+1:
                                    self.exec_action_lists.append(gd)
                                    FLAG=False
                                    break
                                if self.exec_action_lists[index+1]!=gd:
                                    self.exec_action_lists.insert(index+1,gd)
                                    FLAG=False
                                    break
                
                # action before switchon must be close
                if action=='switchon':
                    # check target object restriction
                    if obj not in switchon_obj:
                        self.exec_action_lists.remove(i)
                        FLAG=False
                        break
                    # gd before
                    gd=i.replace('switchon','close')
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
                
                # action before putin must be open, after must be close
                if action=='putin':
                    # check target object restriction
                    if obj2 not in putin_obj:
                        if obj2 in putback_obj:
                            self.exec_action_lists[index]=i.replace('putin','putback')
                            FLAG=False
                            break
                        else:
                            self.exec_action_lists.remove(i)
                            FLAG=False
                            break
                    # action before
                    gd="[{}] <{}> ({})".format('open', obj2, num2)
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
                    # action after
                    gd="[{}] <{}> ({})".format('close', obj2, num2)
                    # not satisfy
                    if len(self.exec_action_lists)<=index+1:
                        self.exec_action_lists.append(gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index+1]!=gd:
                        self.exec_action_lists.insert(index+1,gd)
                        FLAG=False
                        break            
                
                # action before putback must be walk
                if action=='putback':
                    # check target object restriction
                    if obj2 not in putback_obj:
                        if obj2 in putin_obj:
                            self.exec_action_lists[index]=i.replace('putback','putin')
                            FLAG=False
                            break
                        else:
                            self.exec_action_lists.remove(i)
                            FLAG=False
                            break                                              
                    # action before
                    gd="[{}] <{}> ({})".format('walk', obj2, num2)
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break        
                
                # target object restriction of open/close
                if action in ['open','close']:
                    if obj not in oc_obj:
                        self.exec_action_lists.remove(i)      
                        FLAG=False
                        break
                
                # action before open must be walk
                if action=='open':
                    # action before
                    gd="[{}] <{}> ({})".format('walk', obj, num)
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
    
    def get_action_from_llm(self):
        action_obj_str = ''
        if self.exec_action_index >= len(self.exec_action_lists):
            return 'DONE'
        action_obj_str = self.exec_action_lists[self.exec_action_index]
        self.exec_action_index += 1
        return action_obj_str
