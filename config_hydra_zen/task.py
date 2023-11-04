from hydra_zen import store

from tasks import ClassificationTask

task_store = store(group='task')
task_store(
    ClassificationTask,
    num_classes='${num_classes}',
    input_dim=(1, 28, 28)
)
