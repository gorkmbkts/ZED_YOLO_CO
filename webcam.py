from time import time
from time import sleep
start= time()

while True:
    timez = time()
    elaps =timez-start
    
    if elaps>5:
        start=timez
        
    
    print((elaps))
    sleep(1)