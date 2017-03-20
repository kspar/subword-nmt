#!/usr/bin/env python3

import sys

with open(sys.argv[1], encoding='utf-8') as f, open(sys.argv[2], encoding='utf-8') as g, open(sys.argv[3], 'w', encoding='utf-8') as h: 
	list(map(lambda t: h.write(''.join(t) + '\n'), list(zip(f.readlines(), g.readlines()))))

