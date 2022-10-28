import matplotlib.pyplot as plt
import numpy as np


results = []
start_num = 0
end_num = 6
for i in range( end_num):
    data = []
    with open(f'DragLift_case_{i+1}.fce', 'r') as f:
        lines = f.readlines()
        for line in lines[5:]:
            items = line.strip().split()
            items = [float(item) for item in items]
            data.append(items)

    print(data[:3])
    data = np.array(data)
    results.append(data)

# plots
start = [100, 100, 100, 100, 100, 100]
plt.figure(figsize=(5,4))
for i in range(start_num, end_num):
    plt.plot(results[i][start[i]:,0], results[i][start[i]:,6], label=f'case_{i+1}')
plt.grid()
plt.legend()
plt.xlabel('time $c/U_{\inf}$')
plt.ylabel('force dimensionless')
plt.tight_layout()
plt.savefig('lift_convergence.png', dpi=400)
# plt.show()
plt.close()

plt.figure(figsize=(5,4))
for i in range(start_num, end_num):
    plt.plot(results[i][start[i]:,0], results[i][start[i]:,3], label=f'case_{i+1}')
plt.grid()
plt.legend()
plt.xlabel('time $c/U_{\inf}$')
plt.ylabel('force dimensionless')
plt.tight_layout()
plt.savefig('drag_convergence.png', dpi=400)
# plt.show()
plt.close()


# Lift analysis
maxes_all = []
mins_all = []
for i in range(start_num, end_num):
    maxes = []
    mins = []
    for j in range(1,a[i][:,6].shape[0]-1):
        if (results[i][j,6]>results[i][j+1,6]) and (results[i][j,6]>results[i][j-1,6]):
            maxes.append(results[i][j,6])
        if (results[i][j,6]<results[i][j+1,6]) and (results[i][j,6]<results[i][j-1,6]):
            mins.append(results[i][j,6])
    maxes_all.append(maxes)
    mins_all.append(mins)

print('Max lifts: ')
for i in range(start_num, end_num):
    print(' %.4f &' % maxes_all[i][-1], end='')
print()
print('Min lifts: ')
for i in range(start_num, end_num):
    print(' %.4f &' % mins_all[i][-1], end='')
print()
avgs = []
for i in range(start_num, end_num):
    avgs.append(maxes_all[i][-1])
avgs= np.array(avgs).astype(float)
print('Avg errors:')
for i in range(start_num, end_num):
    val = abs((avgs[i]-avgs[end_num-1])/float(avgs[end_num-1])*100)
    print(' %.3f\%% &' % val, end='')
print('\n\n')

# Drag analysis
maxes_all = []
mins_all = []
for i in range(start_num, end_num):
    maxes = []
    mins = []
    for j in range(1,results[i][:,3].shape[0]-1):
        if (results[i][j,3]>results[i][j+1,3]) and (results[i][j,3]>results[i][j-1,3]):
            maxes.append(results[i][j,3])
        if (results[i][j,3]<results[i][j+1,3]) and (results[i][j,3]<results[i][j-1,3]):
            mins.append(results[i][j,3])
    maxes_all.append(maxes)
    mins_all.append(mins)

print('Max drags: ')
for i in range(start_num, end_num):
    print(' %.4f &' % maxes_all[i][-1], end='')
print()
print('Min drags: ')
for i in range(start_num, end_num):
    print(' %.4f &' % mins_all[i][-1], end='')
print()
avgs = []
for i in range(start_num, end_num):
    avgs.append((maxes_all[i][-1]+mins_all[i][-1])/2)
avgs= np.array(avgs).astype(float)
print('maxes errors:')
for i in range(start_num, end_num):
    val = abs((avgs[i]-avgs[end_num-1])/float(avgs[end_num-1])*100)
    print(' %.3f\%% &' % val, end='')
print()