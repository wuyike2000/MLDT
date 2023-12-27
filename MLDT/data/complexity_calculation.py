import os
import pickle

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


path=os.listdir('test_init_env')
for filename in path:
    sum_step=0
    sum_obj=0
    sum_class=0
    max_step=0
    max_obj=0
    max_class=0
    data=pickle.load(open('test_init_env/'+filename,'rb'))
    for sample in data:
        current=compute_task_complexity(sample['task_goal'][0],sample['init_graph'])
        max_step=max(current,max_step)
        sum_step+=current
        # calculate original used obj num
        used_num=0
        # calulate class
        classset=set()
        for i in sample['task_goal'][0].items():
            if 'closed' not in i[0] and 'turnon' not in i[0]:
                used_num+=i[1]
            split=i[0].split('_')
            if len(split)==2:
                classset.add(split[1])
            if len(split)==3:
                classset.add(split[1])
                classset.add(split[2])
        # before not consider put object, this kind of object is represented by numeral in goal
        for i in classset:
            if i.isdigit():
                used_num+=1
        max_obj=max(used_num,max_obj)
        sum_obj+=used_num
        max_class=max(len(classset),max_class)
        sum_class+=len(classset)

    print('Subset:',filename)
    print('Max Step:',max_step)
    print('Average Step:',sum_step/len(data))
    print('Max Obj:',max_obj)
    print('Average Obj:',sum_obj/len(data))
    print('Max Class:',max_class)
    print('Average Class:',sum_class/len(data))