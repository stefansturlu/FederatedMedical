import sys
from datetime import datetime


def logPrint(*msg):
    now = datetime.now()
    time = str(now.strftime("%d/%m/%Y,%H:%M:%S"))
    print(time + ":", end=" ")
    print(*msg)
    sys.stdout.flush()
