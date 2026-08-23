from transformers.integrations import WandbCallback

class HPOWandbCallback(WandbCallback):

    def __init__(self, project_name, group_name):
        super().__init__()
        self.project_name = project_name
        self.group_name = group_name
        self._initialized = False

    def setup(self, args, state, model, **kwargs):

        if self._wandb is None:
            return

        if self._initialized:
            return

        run_name = state.trial_name or args.run_name

        self._wandb.init(
            project=self.project_name,
            group=self.group_name,
            name=run_name,
            config=args.to_dict(),
        )

        self._initialized = True