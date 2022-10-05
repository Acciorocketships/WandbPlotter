from wandb_plotter import Plotter

plotter = Plotter(project="sae-rand-exp")
runs = plotter.group_runs(
		name_func = lambda config: "{group}-{dim}".format(group=config["group"], dim=config["hidden_dim"]),
		filter_config = {"group": lambda group: group=="sae" or group=="transformer"}
	) # a dict of key: groupname to val: list of run objects
groups = plotter.get_data(runs, attr="corr", samples=500) # dict of key: groupname to val: pandas df (index: step, columns: run names)
bounds = plotter.get_bounds(groups, minmax=True) # transformed dict, where the dataframes now have columns "low" "mean" "high"
smoothed = plotter.smooth_data(bounds, smoothing=0.1) # transformed dict, applying some smoothing alpha
ax = plotter.plot(smoothed) # a matplotlib axes object with a minimalistic plot of the given data
pyplot.legend()
pyplot.show()