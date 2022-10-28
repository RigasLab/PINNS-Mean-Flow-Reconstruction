import numpy as np
filename = 'cyl.pts'

x_min = -1
x_max = 3
dx = 0.002
y_min = -1.5
y_max = 1.5
dy = 0.002

x_domain = np.linspace(x_min, x_max, int((x_max-x_min)/dx+1))
y_domain = np.linspace(y_min, y_max, int((y_max-y_min)/dy+1))

with open(filename, 'w+') as f:
    f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
    f.write('<NEKTAR>\n')
    f.write('\t<POINTS DIM="2" FIELDS="">\n')
    for x in x_domain:
        for y in y_domain:
            f.write('\t\t%.3f %.3f\n' % (x, y))
    f.write('\t</POINTS>\n')
    f.write('</NEKTAR>')
