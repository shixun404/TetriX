import ast

def read_dict_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip()
        # Use ast.literal_eval to safely evaluate the string as a Python expression
        my_dict = ast.literal_eval(data)
    return my_dict
perigee = [11, 11, 10, 11, 11, 11, 10, 12, 11, 11, 11, 11, 9, 10, 11, 11, 10, 11, 10, 10]
# Example usage
file_path = 'dict.txt'  # Replace with your file path
result_dict = read_dict_from_txt(file_path)
s = "Latency Instance"
keys = list(result_dict.keys())
key_to_remove = "K_ring_shortest_ring"
if key_to_remove in keys:
    keys.remove(key_to_remove)
for key in keys[:-1]:
    s += f"{key}&"
s += f"{keys[-1]}"
for i in range(20):
    s += f"\nG{i}&{perigee[i]}&"
    for key in keys[:-1]:
        s += f"{result_dict[key][i]:.1f}&"
    s += f"{result_dict[keys[-1]][i]:.1f} \\\\ \hline"
print(s)
    