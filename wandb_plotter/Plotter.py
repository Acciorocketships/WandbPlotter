import wandb
import re
import pandas as pd
from typing import List, Dict, Callable, Any, Optional, Union
from matplotlib import pyplot, cycler


class Plotter:

	def __init__(self, project, user):
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
			"legend.fontsize": 12,
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


	def get_data(self, runs:Dict, attr:str, xattr:str="_step", samples=10000):
		# runs: a dict of key: groupname to val: list of run objects (the output of group_runs)
		# attr: the metric to be fetched
		groups = {}
		for name, runlist in runs.items():
			dflist = []
			for run in runlist:
				df = run.history(samples=samples, keys=[attr], x_axis=xattr, pandas=True)
				if len(df) == 0:
					continue
				if xattr[0] == "_":
					xattr_new = xattr[1:]
				else:
					xattr_new = xattr
				df = df.set_index(df[xattr].rename(xattr_new)).drop(xattr, axis=1)
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


	def plot(self, groups, custom_colour={}, custom_line={}, sort="val"):
		# sort is one of "str" (alphabetical), "val" (ranks based on attr), or list of group names in order
		if sort == "str":
			def natural_keys(text):
				def atoi(text):
					return int(text) if text.isdigit() else text
				return [atoi(c) for c in re.split(r'(\d+)', text)]
			keys = sorted(groups.keys(), key=natural_keys)
		elif sort == "val" or sort == "invval":
			vals = {key: df.mean(axis=1).iloc[-1] for (key, df) in groups.items()}
			last_val = lambda groupname: vals.get(groupname, 0)
			keys = sorted(groups.keys(), key=last_val)
			if sort == "val":
				keys.reverse()
		else:
			keys = sort
		fig, ax = pyplot.subplots()
		minx = float('inf')
		maxx = -float('inf')
		for name in keys:
			data = groups[name]
			if not (("mean" in data.columns) and ("low" in data.columns) and ("high" in data.columns)):
				data = self.get_bounds(data)
			colour = None
			for key, c in custom_colour.items():
				if key in name:
					colour = c
					break
			linestyle = "-"
			for key, l in custom_line.items():
				if key in name:
					linestyle = l
					break
			(mean_line,) = ax.plot(data["mean"], label=name, color=colour, linestyle=linestyle)
			ax.fill_between(data.index, data["low"], data["high"], color=mean_line.get_color(), alpha=0.3)
			minx = min(minx, min(data.index))
			maxx = max(maxx, max(data.index))
		ax.set_xlim(left=minx, right=maxx)
		return ax


	def plot_full(self,
				  attr:str = "episode_reward_mean",
				  name_func:Callable[[Dict[str, Any]], str] = lambda config: config["group"],
				  filter_config:Dict[str, Any] = {},
				  custom_colour:Dict = {},
				  custom_line:Dict = {},
				  samples:int = 1000,
				  sort:Union[str, List[str]] = "val",
				  log:bool = False,
				  smoothing:float = 0,
				  minmax:bool = False,
				  xlabel:Optional[str] = "Step",
				  ylabel:Optional[str] = "Reward",
				  title:Optional[str] = None,
				  ylim:Optional[List[float]]=None,
				  legend:Optional[str] = "best",
				  legend_ncol:int = 1,
				  show:bool = True,
				  tikzsave:bool = False,
				  matplotlibsave:bool = True,
				  savename:str = "",
				  ):
		group_runs = self.group_runs(name_func=name_func, filter_config=filter_config)  # a dict of key: groupname to val: list of run objects
		group_data = self.get_data(group_runs, attr=attr, samples=samples) # dict of key: groupname to val: pandas df (index: step, columns: run names)
		group_bounds = self.get_bounds(group_data, minmax=minmax) # transformed dict, where the dataframes now have columns "low" "mean" "high"
		group_smoothed = self.smooth_data(group_bounds, smoothing=smoothing) # transformed dict, applying some smoothing alpha
		ax = self.plot(group_smoothed, custom_colour=custom_colour, custom_line=custom_line, sort=sort)
		if legend is not None:
			pyplot.legend(loc=legend, ncol=legend_ncol)
		if xlabel is not None:
			pyplot.xlabel(xlabel)
		if ylabel is not None:
			pyplot.ylabel(ylabel)
		if title is not None:
			pyplot.title(title)
		if log:
			pyplot.yscale("log")
		if ylim is not None:
			pyplot.ylim(bottom=ylim[0], top=ylim[1])
		pyplot.grid()
		if len(savename) > 0:
			path = f"{self.project}-{attr}-{savename}"
		else:
			path = f"{self.project}-{attr}"
		if matplotlibsave:
			pyplot.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.1)
		if tikzsave:
			import tikzplotlib
			tikzplotlib.save(f"{path}.tex", textsize=9)
		if show:
			pyplot.show()

