import matplotlib.pyplot as plt
import sys

try:
    f = open(sys.argv[1])
except IOError:
    print 'File', sys.argv[1], 'does not exist.'
    quit()

x = []
y = []

line = f.readline()

while line:
    xy = line.split()
    x.append(float(xy[0]))
    y.append(float(xy[1]))
    line = f.readline()

f.close()

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points from file \'' + sys.argv[1] + '\'')
plt.show()
