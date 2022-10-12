from wandb_plotter import Plotter
from matplotlib import pyplot

plotter = Plotter(project="popgym-edm", user="prorok-lab")
runs = plotter.group_runs(
		name_func = lambda config: "{group}".format(group=config["group"]),
	) # a dict of key: groupname to val: list of run objects
groups = plotter.get_data(runs, attr="episode_reward_mean") # dict of key: groupname to val: pandas df (index: step, columns: run names)
bounds = plotter.get_bounds(groups, minmax=True) # transformed dict, where the dataframes now have columns "low" "mean" "high"
smoothed = plotter.smooth_data(bounds, smoothing=0.1) # transformed dict, applying some smoothing alpha
ax = plotter.plot(smoothed) # a matplotlib axes object with a minimalistic plot of the given data
pyplot.legend()
pyplot.show()