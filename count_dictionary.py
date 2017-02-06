import sys
from collections import Counter

counter = Counter()
with open(sys.argv[1], encoding='utf-8') as f:
    for line in f.readlines():
        for word in line.strip().split():
            counter[word] += 1

print(len(counter))
