# 0.125, 0.125, 0.5, 0.25 -- 8 outcomes
# random number from 1 to 8;

"""

import random

s = ""
b = ""

for i in range(20):
	r = random.randint(1,8)
	if r == 1:
		s+="0"
	elif r == 2:
		s+="1"
	elif r > 6:
		s+="3"
	elif r > 2 and r < 7: 
		s += "2"

print(s)

"""

b=""
s = "32123221312222223302"

for char in s:
	if char == "0":
		b += "000 "
	if char == "1":
		b += "001 "
	if char == "2":
		b += "1 "
	if char == "3":
		b += "01 "

print(b)


# bruh there's a random.choices outcome for this :skull:
