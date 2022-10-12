from wandb_plotter import Plotter

plotter = Plotter(project="sc2-1o_10b_vs_1r")

plotter.plot_full(
	attr="test_battle_won_mean", 
	smoothing=0.6, 
	ylabel="Test Battle Won", 
	name_func=lambda config: "{model} (com range = {com})".format(model=("QMIX-Att" if config["mixer"]=="qmix" else "QGNN"), com=config["comm_range"]),
	custom_colour={"64": "#377eb8", "16": "#ff7f00", "4": "#984ea3", "1": "#e41a1c"},
	custom_line={"QGNN": '-', "QMIX-Att": "--"},
	legend="lower center",
	legend_ncol=2,
	samples=500,
)