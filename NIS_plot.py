import csv
import matplotlib.pyplot as plt
import numpy as np

nis_lidar = []
nis_radar = []

with open('data/output.txt', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    reader = csv.DictReader(csvfile,delimiter=',')
    for row in reader:
        if row[' sensor'] == " L":
            nis_lidar.append(float(row[' nis']))
        else:
            nis_radar.append(float(row[' nis']))


chi_radar = 7.8* np.ones(len(nis_radar))
t_r = range(len(nis_radar))

chi_lidar=5.99 * np.ones(len(nis_lidar))
t_l = range(len(nis_lidar))



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
ax1.set_title('NIS radar')
ax1.plot(t_r, chi_radar,color='r')
ax1.plot(t_r, nis_radar,color='k')
ax2.set_title('NIS lidar')
ax2.plot(t_l, chi_lidar,color='r')
ax2.plot(t_l, nis_lidar,color='k')

plt.show()
plt.savefig('NIS.png')