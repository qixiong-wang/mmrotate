import torch
import torch.nn as nn
import torch.distributed as dist



class Large_batch_queue_classwise(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_classes=37, number_of_instance=2, feat_len=256):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(Large_batch_queue_classwise, self).__init__()
        self.num_classes=37
        self.register_buffer("large_batch_queue", torch.zeros(num_classes, number_of_instance, feat_len))
        self.register_buffer("tail", torch.zeros(num_classes).long())


    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        if features.get_device() == 0:
            import pdb
            pdb.set_trace()
        else:
            dist.barrier()
        with torch.no_grad():
            for indx, label in enumerate(torch.unique(pid_labels)):
                if label >= 0 and label<self.num_classes:
                    self.large_batch_queue[label,self.tail[label]] = torch.mean(features[pid_labels==label],dim=0)
                    self.tail[label]+=1
                    if self.tail[label] >= self.large_batch_queue.shape[1]:
                        self.tail[label] -= self.large_batch_queue.shape[1]
        return self.large_batch_queue