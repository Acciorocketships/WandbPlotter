from wandb_plotter import Plotter

plotter = Plotter(project="sae-rand-exp")
plotter.plot_full(attr="corr", smoothing=0.1, ylabel="Correlation", filter_config={"hidden_dim": 96})