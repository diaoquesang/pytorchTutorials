import time

file = open('log.txt', 'w')  # 保存日志位置

# 通过给print函数的file赋值,可以将print的内容转而输出到file文件中,便于后台实时监控日志
print(time.strftime("%H:%M:%S", time.localtime()), "Hello World!", file=file)
time.sleep(1)
print(time.strftime("%H:%M:%S", time.localtime()), "Hello Amy!", file=file)
print(time.strftime("%H:%M:%S", time.localtime()), "Hello Bob!", file=file)
print(time.strftime("%H:%M:%S", time.localtime()), "Hello Cindy!", file=file)
print(time.strftime("%H:%M:%S", time.localtime()), "Hello Daddy!", file=file)
