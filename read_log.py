res = []
with open("output.log", 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "test_log/rapid" in line:
            print(line[-6:-2])
            res.append(int(line[-6:-2]))

for i in range(200):
    print(i + 1234, res[i])
    if res[i] != 1234 + i:
        assert 0
