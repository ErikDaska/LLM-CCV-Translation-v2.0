from transformers.integrations import WandbCallback

class HPOWandbCallback(WandbCallback):

    def __init__(self, project_name, group_name):
        super().__init__()
        self.project_name = project_name
        self.group_name = group_name

    def setup(self, args, state, model, **kwargs):

        if self._wandb is None:
            return

        # During hyperparameter search, HF stores the trial name here.
        run_name = state.trial_name

        if run_name is None:
            run_name = args.run_name

        self._wandb.init(
            project=self.project_name,
            group=self.group_name,
            name=run_name,
            config=args.to_dict(),
            reinit=True,
        )