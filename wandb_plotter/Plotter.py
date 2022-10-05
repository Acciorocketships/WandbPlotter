import wandb
import pandas as pd
from typing import List, Dict, Callable, Any, Optional
from matplotlib import pyplot, cycler
import tikzplotlib


class Plotter:

	def __init__(self, project, user="prorok-lab"):
		self.project = project
		self.user = user
		self.api = wandb.Api(timeout=20)
		self.fetch_runs()
		self.pyplot_setup()


	def pyplot_setup(self):
		tex_fonts = {
			# Use LaTeX to write all text
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Times"],
			# Use 10pt font in plots, to match 10pt font in document
			"axes.labelsize": 16,
			"font.size": 25,
			# Make the legend/label fonts a little smaller
			"legend.fontsize": 16,
			"legend.title_fontsize": 7,
			"legend.framealpha": 0.3,
			"xtick.labelsize": 16,
			"ytick.labelsize": 16,
			# Figure Size
			"figure.figsize": (10, 5),
			# Colour Cycle
			"axes.prop_cycle": cycler(color=["#377eb8","#ff7f00","#4daf4a","#f781bf","#a65628",
									  		 "#984ea3","#2bcccc","#999999","#e41a1c","#dede00"])

		}
		pyplot.rcParams.update(tex_fonts)


	def update_project(self, project):
		self.project = project
		self.fetch_runs()


	def fetch_runs(self):
		self.runs = self.api.runs("{user}/{project}".format(user=self.user, project=self.project))
		return self.runs


	def group_runs(self, 
				   name_func:Callable[[Dict[str, Any]], str] = lambda config: config["group"],
				   filter_config:Dict[str, Any] = {}):
		# name_func: A function which outputs the name for a group of runs. If two runs have the same name, they are in the same group
		#            The input to the function is a dict of keys from group_config and their corresponding values for each run
		# filter_config: A dictionary of config keys and values. If provided, it will filter out any runs which do not match
		# 			 If the provided value is a function, then it will apply that function as a filter (ex: {"step": lambda x: x > 100})
		# output: A dict mapping strings (the output of name_func) to lists of runs
		# config entries: "group", "name", "state", "step", "timestamp", "runtime", all logged metrics at the last step, all entries from the config passed to wandb.init

		groups = {}

		for run in self.runs:
			run_config = run.config
			run_config["group"] = run.group
			run_config["state"] = run.state
			run_config["name"] = run.name
			run_summary = {(key if key[0]!="_" else key[1:]): val for (key, val) in run.summary.items() if (isinstance(val, int) or isinstance(val, float))}
			run_config.update(run_summary)

			# Filter
			skip = False
			for filter_key, filter_val in filter_config.items():
				if isinstance(filter_val, Callable):
					if not filter_val(run_config[filter_key]):
						skip = True
						break
				else:
					if run_config[filter_key] != filter_val:
						skip = True
						break
			if skip:
				continue

			# Grouping
			name = name_func(run_config)
			if name not in groups:
				groups[name] = []
			groups[name].append(run)

		return groups


	def get_data(self, runs:Dict, attr:str, samples=500):
		# runs: a dict of key: groupname to val: list of run objects (the output of group_runs)
		# attr: the metric to be fetched
		groups = {}
		for name, runlist in runs.items():
			dflist = []
			for run in runlist:
				df = run.history(samples=samples, keys=[attr], x_axis="_step", pandas=True)
				if len(df) == 0:
					continue
				df = df.set_index(df["_step"].rename("step")).drop("_step", axis=1)
				df = df.rename(columns={attr: run.name})
				dflist.append(df)
			cat_nan = pd.concat(dflist, sort=True, axis=1)
			groups[name] = self.interpolate_data(cat_nan)
		return groups


	def interpolate_data(self, df):
		return df.interpolate(method='index', limit_area="inside", axis=0)


	def smooth_data(self, df, smoothing=0):
		if isinstance(df, dict):
			return {key: self.smooth_data(x, smoothing=smoothing) for key, x in df.items()}
		halflife = len(df) / 100 * smoothing
		return df.ewm(halflife=halflife).mean()


	def get_bounds(self, df, minmax=False):
		if isinstance(df, dict):
			return {key: self.get_bounds(x) for key, x in df.items()}
		mean = df.mean(axis=1)
		if minmax:
			low = df.min(axis=1)
			high = df.max(axis=1)
		else:
			std = df.std(axis=1)
			low = mean - std
			high = mean + std
		bounds = pd.concat([low, mean, high], keys=["low", "mean", "high"], axis=1)
		return bounds


	def plot(self, groups, custom_colour={}):
		fig, ax = pyplot.subplots()
		for name, data in groups.items():
			if not (("mean" in data.columns) and ("low" in data.columns) and ("high" in data.columns)):
				data = self.get_bounds(data)
			(mean_line,) = ax.plot(data["mean"], label=name, color=custom_colour.get(name, None))
			ax.fill_between(data.index, data["low"], data["high"], color=mean_line.get_color(), alpha=0.3)
		return ax


	def plot_full(self,
				  attr:str = "episode_reward_mean",
				  name_func:Callable[[Dict[str, Any]], str] = lambda config: config["group"],
				  filter_config:Dict[str, Any] = {},
				  custom_colour:Dict = {},
				  samples:int = 1000,
				  smoothing:float = 0,
				  minmax:bool = False,
				  xlabel:Optional[str] = "Step",
				  ylabel:Optional[str] = "Reward",
				  title:Optional[str] = None,
				  legend:Optional[str] = "best",
				  show:bool = True,
				  tikzsave:bool = False,
				  matplotlibsave:bool = True,
				  savename:str = "",
				  ):
		group_runs = self.group_runs(name_func=name_func, filter_config=filter_config)  # a dict of key: groupname to val: list of run objects
		group_data = self.get_data(group_runs, attr=attr, samples=samples) # dict of key: groupname to val: pandas df (index: step, columns: run names)
		group_bounds = self.get_bounds(group_data, minmax=minmax) # transformed dict, where the dataframes now have columns "low" "mean" "high"
		group_smoothed = self.smooth_data(group_bounds, smoothing=smoothing) # transformed dict, applying some smoothing alpha
		ax = self.plot(group_smoothed, custom_colour=custom_colour)
		if legend is not None:
			pyplot.legend(loc=legend)
		if xlabel is not None:
			pyplot.xlabel(xlabel)
		if ylabel is not None:
			pyplot.ylabel(ylabel)
		if title is not None:
			pyplot.title(title)
		pyplot.grid()
		if len(savename) > 0:
			path = f"{self.project}-{attr}-{savename}"
		else:
			path = f"{self.project}-{attr}"
		if matplotlibsave:
			pyplot.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0)
		if tikzsave:
			tikzplotlib.save(f"{path}.tex", textsize=9)
		if show:
			pyplot.show()




def example1():
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
	breakpoint()


def example2():
	plotter = Plotter(project="sae-rand-exp")
	plotter.plot_full(attr="corr", smoothing=0.1, ylabel="Correlation", filter_config={"hidden_dim": 96})


if __name__ == '__main__':
	example2()
