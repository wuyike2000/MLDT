import os
import random
import pickle
from tqdm import tqdm

random.seed(23)
# setting
max_num=6
com_threshold=60

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

grab_obj=['pancake','poundcake','cutleryfork','milk','wineglass','waterglass','plate','chicken','cupcake']
put_obj=['fridge','dishwasher','stove','kitchentable','microwave','sink']
switchon_obj=['stove','microwave','dishwasher']
on_obj=['sink','kitchentable']
oc_obj=['fridge']
loc_obj=['fridge','dishwasher','stove','kitchentable','microwave','sink','kitchencabinet','cabinet','livingroom','kitchen','bedroom']

path=os.listdir('test_init_env')
longset=[]

num=0
for filename in path:
    data=pickle.load(open('test_init_env/'+filename,'rb'))
    for sample in tqdm(data):
        # record grab obj num
        grabdict=dict()
        # record put obj id
        putdict=dict()
        # record grab abj loc
        locdict=dict()
        # collect grab obj num, put obj id
        for node in sample['init_graph']['nodes']:
            # collect grab obj
            if node['class_name'] in grab_obj:
                # find location
                for edge in sample['init_graph']['edges']:
                    if edge['from_id'] == node['id'] and edge['relation_type'] == 'INSIDE':
                        location=[node['class_name'] for node in sample['init_graph']['nodes'] if node['id'] == edge['to_id']][0]
                        # filter new location
                        if location in loc_obj:
                            # record obj
                            if not grabdict.get(node['class_name']):
                                grabdict[node['class_name']]=0
                            grabdict[node['class_name']]+=1
                            # record obj location
                            if not locdict.get(location):
                                locdict[location]=set()
                            locdict[location].add(edge['to_id'])
                            break
                
            # collect put obj
            if node['class_name'] in put_obj:
                if not putdict.get(node['class_name']):
                    putdict[node['class_name']]=set()
                putdict[node['class_name']].add(node['id'])    
           
        # filter putdict exclude grab obj location
        for i in locdict.items():
            if putdict.get(i[0]):
                for j in i[1]:
                    putdict[i[0]].remove(j)
        putdict = {key: value for key, value in putdict.items() if len(value) > 0}

        # calculate original used obj num
        used_num=0
        for i in sample['task_goal'][0].items():
            if 'closed' not in i[0] and 'turnon' not in i[0]:
                used_num+=i[1]
        
        # calculate all obj num
        obj_num=0
        for i in grabdict.values():
            obj_num+=i
            
        # new goal
        new_goal=dict()

        # consider extend more than 5 objs unused
        if obj_num-used_num>5:
            # put obj list
            putlist=list(putdict.keys())
            # iterate all obj
            for i in grabdict.items():
                # random choice put obj
                # if no put obj can choose
                if len(putlist)==0:
                    break
                target_put=random.choice(putlist)
                target_put_id=list(putdict[target_put])[0]
                putlist.remove(target_put)
                # target_num cannot more than max_num
                target_num=min(max_num,i[1])
                if target_put in switchon_obj:
                    new_goal['closed_'+str(target_put_id)]=1
                    new_goal['turnon_'+str(target_put_id)]=1
                    new_goal['inside_'+i[0]+'_'+str(target_put_id)]=target_num
                if target_put in on_obj:
                    new_goal['on_'+i[0]+'_'+str(target_put_id)]=target_num
                if target_put in oc_obj:
                    new_goal['closed_'+str(target_put_id)]=1
                    new_goal['inside_'+i[0]+'_'+str(target_put_id)]=target_num                    
            
            # judge the complexity is over threshold
            if compute_task_complexity(new_goal,sample['init_graph'])>com_threshold:
                # modify task id
                sample['task_id']=num
                new_task_goal=dict()
                new_task_goal[0]=new_goal
                new_task_goal[1]=new_goal
                # modify task goal
                sample['task_goal']=new_task_goal
                longset.append(sample)

                num+=1
                        
                print(sample['task_goal'])        
                print('complexity:',compute_task_complexity(new_goal,sample['init_graph']))

pickle.dump(longset,open('test_init_env/LongTasks.p','wb'))
print(num)
        

                
