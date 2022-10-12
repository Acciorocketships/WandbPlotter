from wandb_plotter import Plotter

plotter = Plotter(project="sae-rand-exp")

plotter.plot_full(
	attr="corr", 
	smoothing=0.1, 
	ylabel="Correlation", 
	name_func=lambda x: {"sae": "SAE", "rnn": "GRU", "dspn": "DSPN", "tspn": "TSPN"}.get(x["group"], x["group"]),
	filter_config={"hidden_dim": 96, "group": lambda x: x != "transformer"}, 
	legend="upper right", 
	sort=["SAE", "GRU", "DSPN", "TSPN"],
	ylim=[0., 1.],
	show=False,
	savename="96",
)

plotter.plot_full(
	attr="mse_loss", 
	smoothing=0.1, 
	ylabel="Mean Squared Error",
	name_func=lambda x: {"sae": "SAE", "rnn": "GRU", "dspn": "DSPN", "tspn": "TSPN"}.get(x["group"], x["group"]),
	filter_config={"hidden_dim": 96, "group": lambda x: x != "transformer"}, 
	legend="upper right",
	log=True, 
	sort="invval",
	show=False,
	savename="96",
)

plotter.plot_full(
	attr="corr", 
	smoothing=0.1, 
	ylabel="Correlation", 
	name_func=lambda x: {"sae": "SAE", "rnn": "GRU", "dspn": "DSPN", "tspn": "TSPN"}.get(x["group"], x["group"]),
	filter_config={"hidden_dim": 48, "group": lambda x: x != "transformer"}, 
	legend="upper right", 
	sort=["SAE", "GRU", "DSPN", "TSPN"],
	ylim=[0., 1.],
	show=False,
	savename="48",
)

plotter.plot_full(
	attr="mse_loss", 
	smoothing=0.1, 
	ylabel="Mean Squared Error", 
	name_func=lambda x: {"sae": "SAE", "rnn": "GRU", "dspn": "DSPN", "tspn": "TSPN"}.get(x["group"], x["group"]),
	filter_config={"hidden_dim": 48, "group": lambda x: x != "transformer"}, 
	legend="upper right",
	log=True, 
	sort="invval",
	ylim=[0.09, 1.1],
	show=False,
	savename="48",
)