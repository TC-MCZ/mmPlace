import matplotlib.pyplot as plt
import numpy as np

odometry_file = open("../sequence0/raw_data/radar_odometry.txt")

odometry_list = []
for pose in odometry_file:
    pose = pose.split(" ")
    odometry_list.append([float(pose[2]), float(pose[3]), float(pose[4])])
odometry = np.array(odometry_list)
print(odometry.shape)

odometry_file.close()

fig, axs = plt.subplots(1, 1)

axs.scatter(odometry[:, 0], odometry[:, 1])
axs.tick_params(labelsize=20)
axs.set_xlabel('X(m)', fontsize=20)
axs.set_ylabel('Y(m)', fontsize=20)

plt.show()

