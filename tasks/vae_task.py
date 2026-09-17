from tasks.base_task import Task

class VAE(Task):
    def __init__(self, **kwargs):
        super().__init__('both')
