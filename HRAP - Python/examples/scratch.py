import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2], [3, 4])
btn_ax = ax.inset_axes([0.8, 0.8, 0.1, 0.1])
print(len(ax.child_axes))
ax.clear()
print(len(ax.child_axes))
