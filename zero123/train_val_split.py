import os

target_dir = '/mnt/datassd/seeha/data/3D/objaverse'
objects = os.listdir(target_dir)

print(len(objects))

train_txt = open(f'train.txt', 'w')
val_txt = open(f'val.txt', 'w')

for obj in objects:
    obj_path = os.path.join(target_dir, obj)
    obj_list = os.listdir(obj_path)
    obj_list.remove('prompts.json')
    remove_list = []
    
    for i in obj_list:
        dir_path = os.path.join(obj_path, i, 'annotation')
        num = len(os.listdir(dir_path))
        if num < 2:
            remove_list.append(i)
    
    for i in remove_list:
        obj_list.remove(i)
    
    obj_num = len(obj_list)
    train_num = int(obj_num*0.8)
    val_num = obj_num - train_num
    for i, o in enumerate(obj_list):
        if i < train_num:
            train_txt.write(f'{obj}/{o}\n')
        else:
            val_txt.write(f'{obj}/{o}\n')
            
    
    
    
    # print(train_num)
    
    # print(len(os.listdir(obj_path))//2)
    # print
    # if len(os.listdir(obj_path)) <= 3:
    #     print(len(os.listdir(obj_path)))

# for obj in objects:
#     obj_path = os.path.join('train', obj)
#     for i in os.listdir(obj_path):
#         print(len(os.listdir()))
    #     if not '.' in i:
    #         f.write(f'{obj}/{i}\n')
    # print(len(os.listdir(obj_path)))

# with open('/mnt/datassd/seeha/data/3D/train.txt') as f:
#     rel_paths = f.read().splitlines()

# paths = [os.path.join('a', i) for i in rel_paths]
# print(paths)