
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.api.registry import MODEL_REGISTRY
from peft import  PeftModel
import torch

"""
TO BE ABLE TO USE THIS
(first try to use as is, if it fails try the following):

$ git clone https://github.com/EleutherAI/lm-evaluation-harness 
$ cd lm-evaluation-harness
$ git reset --hard 4d7d2f64576205105318fd12a622b6f0b7c70464
$ pip install -e .
"""

# default tasks
TASKS = [
    # common sense reasoning
    "winogrande", # https://huggingface.co/datasets/winogrande
    "boolq", # https://huggingface.co/datasets/google/boolq
    
    # language understanding and knowledge
    "piqa", # https://huggingface.co/datasets/piqa
    "hellaswag", # https://huggingface.co/datasets/Rowan/hellaswag
]

class TaskManagerValid(TaskManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_task_valid_only(self, task):
        """ Overrides tasks which evaluate on the dataset s.t. they do evaluation on
        the evlauation set instead.
        """
        def has_test_docs():
            return False
        def test_docs():
            return None
        def has_validation_docs():
            return False
        def validation_docs():
            return None
        def training_docs():
            import pdb; pdb.set_trace()
        task.has_test_docs = has_test_docs
        task.test_docs = test_docs
        task.has_validation_docs = has_validation_docs
        task.validation_docs = validation_docs
    
    def load_task_or_group(self, task_list) -> dict:
        task_dict = super().load_task_or_group(task_list)
        for task in task_dict.values():
            self.make_task_valid_only(task)
        return task_dict

@torch.no_grad()
def evaluate_on_nlp_tasks(
    model,
    tokenizer,
    max_length = 1024, # max context length
    batch_size = 1,
    tasks = None, # use TASKS by default
    limit = None, # if int: number of samples per task, if float < 1: percentage of examples in task(dataset)
    return_samples = False,
    bootstrap_iters = 0, # number of bootrstrap iterations to compute statistics
    verbosity = "ERROR",
):
    """ This function will use the evaluation set of the datasets to do evaluation.
    """
    was_training = model.training
    lm_model = MODEL_REGISTRY["hf"](
        model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        max_batch_size=batch_size,
        trust_remode_code=True,
    )

    tasks = tasks if tasks is not None else TASKS

    res = evaluator.simple_evaluate(
        model=lm_model,
        tasks = tasks,
        limit=limit,
        log_samples=return_samples, # do not want to get samples in output for efficiency
        bootstrap_iters=bootstrap_iters, # avoid bootstrap statistics for efficiency
        task_manager=TaskManagerValid(),
        verbosity=verbosity,
    )
    model.train(was_training)
    return res

def evaluate_checkpoint(base_model, checkpoint, tokenizer, tasks=None, limit=None):
    """
    base_model: model after pruning/before posttraining
    checkpoint: path to training checkpoint with lora weights
    tokenizer: tokenizer for model
    tasks: list of evaluation tasks, default uses global TASKS
    limit:
        if None: evaluate on entire dataset
        if int: number of samples per task 
        if float < 1: percentage of examples in task(dataset)
    """
    model = PeftModel.from_pretrained(base_model, checkpoint)
    return evaluate_on_nlp_tasks(model, tokenizer, tasks=tasks, limit=limit)
