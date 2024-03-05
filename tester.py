import ujson as json  # Import json for dictionary serialization

with open('qtable_navigation.json') as table_navi:
    q_table_navigation = json.load(table_navi)
print(q_table_navigation)