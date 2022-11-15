
if __name__ == '__main__':
  old_graph = {}
  id_map = {}
  node_num = 65608366
  path = '/data/com-friendster.ungraph.txt'
  with open(path, 'r') as f:
    head = 4
    id_cnt = 0
    for line in f:
      if head > 0:
        head -= 1
        continue
      id1, id2 = line.split()
      id1 = int(id1)
      id2 = int(id2)
      if id1 not in old_graph.keys():
        old_graph[id1] = []
      old_graph[id1].append(id2)
      if id1 not in id_map.keys():
        id_map[id1] = id_cnt
        id_cnt += 1
      if id2 not in id_map.keys():
        id_map[id2] = id_cnt
        id_cnt += 1
  
  new_graph = {}
  for src in old_graph.keys():
    new_src = id_map[src]
    new_graph[new_src] = []
    for des in old_graph[src]:
      new_graph[new_src].append(str(id_map[des]))
  
  new_path = '/data/com-friendster.ungraph.compact.txt'
  with open(new_path, 'w') as f:
    line = str(node_num) + '\n'
    f.write(line)
    for id in range(node_num):
      if id not in new_graph.keys():
        line = '\n'
      else:
        line = ' '.join(new_graph[id]) + '\n'
      f.write(line)

  # new_graph = {}
  # new_graph[0] = []
  # new_graph[0].append(str(1))
  # new_graph[0].append(str(2))
  # new_graph[1] = []
  # new_graph[1].append(str(2))
  # new_graph[1].append(str(3))
  # new_graph[1].append(str(4))
  # new_graph[1].append(str(5))
  # new_path = '/data/snap/com-friendster.ungraph.compact.txt'

  # with open(new_path, 'w') as f:
  #   for i in range(2):
  #     line = ' '.join(new_graph[i]) + '\n'
  #     print(line)
  #     f.write(line)
