import sys

PKTGEN_FILE = sys.argv[1]
RATE = sys.argv[2]
LINE_IDENTIFICATOR = "RatedUnqueue("

with open(PKTGEN_FILE, 'r') as f:
    lines = f.readlines()

# Count the number of line containing the identificator
count = 0
for line in lines:
    if LINE_IDENTIFICATOR in line:
        count += 1

PER_CORE_RATE = int(float(RATE) / count)

for i, line in enumerate(lines):
    if LINE_IDENTIFICATOR in line:
        splitted_line = line.split(LINE_IDENTIFICATOR)
        params = splitted_line[1].split(",")
        params[0] = str(PER_CORE_RATE)
        # Rebuild the line
        new_line = splitted_line[0] + LINE_IDENTIFICATOR + ",".join(params)
        # Replace the line
        lines[i] = new_line

# Rewrite the file
with open(PKTGEN_FILE, 'w') as f:
    f.writelines(lines)