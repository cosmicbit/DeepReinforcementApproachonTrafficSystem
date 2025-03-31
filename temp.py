log=open("log.txt","w")

print("hello world")
for i in range(2**120):
    log.write(str(i))

