# from enum import Enum
# from ensembler import dataset_statistics
# from ensembler import dataset_initialize
# from ensembler import evaluate
# from ensembler import visualize
from ensembler import monai_init

monai_init.initialize()

# class Tasks(Enum):
#     split = "split"
#     dataset_initialize = "initialize"
#     evaluate = "evaluate"
#     visualize = "visualize"

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return str(self)

#     @classmethod
#     def choices(cls):
#         return sorted([e.value for e in cls])

#     def description(task):
#         if task == "evaluate":
#             return evaluate.description
#         if task == "visualize":
#             return visualize.description
#         if task == "split":
#             return dataset_statistics.description
#         if task == "initialize":
#             return dataset_initialize.description

#         raise ValueError("Task {} not defined".format(task))

#     def add_argparse_args(task):
#         if task == "evaluate":
#             return evaluate.add_argparse_args
#         if task == "visualize":
#             return visualize.add_argparse_args
#         if task == "split":
#             return dataset_statistics.add_argparse_args
#         if task == "initialize":
#             return dataset_initialize.add_argparse_args

#         raise ValueError("Task {} not defined".format(task))

#     def get(task):
#         if task == "evaluate":
#             return evaluate.execute
#         if task == "visualize":
#             return visualize.execute
#         if task == "split":
#             return dataset_statistics.execute
#         if task == "initialize":
#             return dataset_initialize.execute

#         raise ValueError("Task {} not defined".format(task))
