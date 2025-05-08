import numpy as np
import matplotlib.pyplot as plt

# Given data
labels = ['70', '50', '20']
data = [
    [4.773715972900391, 41.84448719024658, 6.368489503860474, 44.692301988601685, 100.0109748840332, 65.39255857467651, 2.7163093090057373, 2.716402053833008],
    [2.353402853012085, 33.25324773788452, 8.948792457580566, 3.5905327796936035, 9.063864707946777, 2.7439112663269043, 1.8899083137512207, 4.1344897747039795],
    [9.233639001846313, 6.678992748260498, 26.76935648918152, 54.65228605270386, 4.836617708206177, 3.4193596839904785, 2.4515914916992188, 7.517486333847046]
]

# Compute averages
averages = [np.mean(arr) for arr in data]

# Plot bar graph with updated labels and title
plt.figure(figsize=(5, 3))
plt.bar(labels, averages, color=['blue', 'orange', 'red'])
plt.xlabel('Initial Demo Length')
plt.ylabel('Average Time')
plt.title('Average Time to Complete the Course')
plt.ylim(0, max(averages) + 5)
plt.tight_layout()
# Display the plot
plt.show()
