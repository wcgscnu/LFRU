import os
import shutil

T_table = ['2.200', '2.220', '2.240', '2.260', '2.280',
           '2.300', '2.320', '2.340', '2.360', '2.380', '2.400']
num_values = len(T_table)

Tc = 2.27
root = './data_Ising120/'
for rt, dirs, files in os.walk(root):
    for file in files:
        temp = file.split('Ising_L120_T')[1].split('_SEED')
        T = float(temp[0])
        SEED = int(temp[1].split('.png')[0])
        print(file, T, SEED)

        if T < Tc:
            if SEED < 500:
                os.rename(os.path.join(rt, file), os.path.join(root, 'train/A', file))
            elif SEED < 600:
                os.rename(os.path.join(rt, file), os.path.join(root, 'val/A', file))
            elif SEED < 700:
                os.rename(os.path.join(rt, file), os.path.join(root, 'test/A', file))

        else:
            if SEED < 500:
                os.rename(os.path.join(rt, file), os.path.join(root, 'train/B', file))
            elif SEED < 600:
                os.rename(os.path.join(rt, file), os.path.join(root, 'val/B', file))
            elif SEED < 700:
                os.rename(os.path.join(rt, file), os.path.join(root, 'test/B', file))


