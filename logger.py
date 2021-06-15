import sys
from datetime import datetime


def logPrint(*msg):
    """
    Prints messages with time of occurrence down to the second
    """
    now = datetime.now()
    time = str(now.strftime("%d/%m/%Y,%H:%M:%S"))
    print(time + ":", end=" ")
    print(*msg)
    sys.stdout.flush()
