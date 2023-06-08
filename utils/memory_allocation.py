import torch
import math
import heapq
from typing import Tuple


class IndexTracker:
    def __init__(self, debug=False):
        self.added_back = []
        self.current = 0
        self.debug = debug
        if self.debug:
            self.removed_set = set()

    def pop_next(self) -> int:
        current = self.current
        if len(self.added_back) > 0 and self.added_back[0] < current:
            smallest = heapq.heappop(self.added_back)
            if self.debug:
                self.removed_set.add(smallest)
            return smallest
        else:
            self.current += 1
            if self.debug:
                self.removed_set.add(current)
            return current

    def remove(self, idx: int):
        if self.debug and idx not in self.removed_set:
            raise RuntimeError("Attempted to remove an item twice")
        if self.debug:
            self.removed_set.remove(idx)
        heapq.heappush(self.added_back, idx)
        return True


class BatchedIndexTracker:
    def __init__(self, batch_size, debug=False):
        self.trackers = []
        for _ in range(batch_size):
            self.trackers.append(IndexTracker(debug=debug))
        self.next_indices = torch.zeros((batch_size,), dtype=torch.long)

    def pop_next(self) -> Tuple[torch.LongTensor, int]:
        max_value = 0
        for i, tracker in enumerate(self.trackers):
            idx = tracker.pop_next()
            self.next_indices[i] = idx
            max_value = max(max_value, idx)
        return self.next_indices, max_value

    def remove(self, batch_indices, positions):
        # print(len(batch_indices))
        for i in range(len(batch_indices)):
            # print("REMOVING", batch_indices[i], positions[i])
            self.trackers[batch_indices[i]].remove(positions[i])


class DynamicTensorFast:
    def __init__(
        self,
        batch_size,
        embedding_dims,
        capacity=64,
        resizable=True,
        reduce_fragmentation=False,
        compact=True,
        dtype=torch.float32,
        device="cpu",
        debug=False,
    ):
        self.batch_size = batch_size
        self.capacity = capacity
        self.embedding_dims = embedding_dims
        self.resizable = resizable
        self.reduce_fragmentation = reduce_fragmentation
        self.debug = debug
        self.compact = compact

        self.tensors = []
        for i, embedding_dim in enumerate(embedding_dims):
            if isinstance(embedding_dim, (list, tuple)):
                # Number of heads is specified
                assert len(embedding_dim) == 2
                self.tensors.append(
                    torch.zeros(
                        (batch_size, embedding_dim[0], capacity, embedding_dim[1]),
                        dtype=dtype,
                        device=device,
                    )
                )  # !!!
            else:
                self.tensors.append(
                    torch.zeros(
                        (batch_size, capacity, embedding_dim),
                        dtype=dtype,
                        device=device,
                    )
                )

        self.mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        self.max_padded_length = 0

        if self.debug:
            self.token_ids = torch.full(
                (batch_size, capacity), dtype=torch.long, device=device, fill_value=-1
            )
            self.next_token_id = 0

    def to(self, device=None, dtype=None):
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device=device, dtype=dtype)
        self.mask = self.mask.to(device=device)
        if self.debug:
            self.token_ids = self.token_ids.to(device=device)

    def _effective_size(self):
        if self.reduce_fragmentation and self.max_padded_length > 0:
            return 2 ** int(math.ceil(math.log2(self.max_padded_length)))
        else:
            return self.max_padded_length

    def _resize(self, new_capacity):
        for i, old_tensor in enumerate(self.tensors):
            if len(old_tensor.shape) == 4:
                new_tensor = torch.zeros(
                    (
                        old_tensor.shape[0],
                        old_tensor.shape[1],
                        new_capacity,
                        old_tensor.shape[3],
                    ),
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
            else:
                new_tensor = torch.zeros(
                    (old_tensor.shape[0], new_capacity, old_tensor.shape[2]),
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
            new_tensor[..., : self.capacity, :] = old_tensor[..., : self.capacity, :]
            self.tensors[i] = new_tensor

        new_mask = torch.zeros(
            (self.mask.shape[0], new_capacity),
            dtype=self.mask.dtype,
            device=self.mask.device,
        )
        new_mask[:, : self.capacity] = self.mask[:, : self.capacity]
        self.mask = new_mask

        if self.debug:
            new_token_ids = torch.full(
                (self.token_ids.shape[0], new_capacity),
                dtype=self.token_ids.dtype,
                device=self.token_ids.device,
                fill_value=-1,
            )
            new_token_ids[:, : self.capacity] = self.token_ids[:, : self.capacity]
            self.token_ids = new_token_ids

        self.capacity = new_capacity

    def append(self, tensors) -> torch.LongTensor:
        # Sanity check
        assert len(tensors) == len(self.embedding_dims)
        for tensor, embedding_dim in zip(tensors, self.embedding_dims):
            if isinstance(embedding_dim, (tuple, list)):
                # Number of heads is specified
                assert len(tensor.shape) == 3
                assert tensor.shape[0] == self.batch_size
                assert tensor.shape[1] == embedding_dim[0]
                assert tensor.shape[2] == embedding_dim[1]
            else:
                assert len(tensor.shape) == 2
                assert tensor.shape[0] == self.batch_size
                assert tensor.shape[1] == embedding_dim

        # Find insertion point
        effective_size = self._effective_size()
        if effective_size == 0:
            max_length = 0
            self.max_padded_length = 1
            insertion_point = torch.zeros(
                (self.batch_size,), device=self.mask.device, dtype=torch.long
            )
        else:
            mask = self.mask[:, :effective_size]
            result = mask.min(dim=1)
            insertion_point = (
                result.indices * (~result.values) + mask.shape[1] * result.values
            )
            max_length = insertion_point.max().item()
            self.max_padded_length = max(self.max_padded_length, max_length + 1)

        if max_length == self.capacity:
            # Needs resizing
            if not self.resizable:
                raise RuntimeError(
                    "The pre-allocated buffer has been exhausted. "
                    "Increase the capacity or set resizable=True."
                )
            new_capacity = (self.capacity * 2) if self.capacity > 0 else 1
            self._resize(new_capacity)

        for i, tensor in enumerate(tensors):
            if len(tensor.shape) == 3:
                self.tensors[i].scatter_(
                    2,
                    insertion_point[:, None, None, None].expand(
                        -1, tensor.shape[1], -1, tensor.shape[-1]
                    ),
                    tensor[:, :, None],
                )
            else:
                self.tensors[i].scatter_(
                    1,
                    insertion_point[:, None, None].expand(-1, -1, tensor.shape[-1]),
                    tensor[:, None],
                )

        self.mask.scatter_(1, insertion_point[:, None], True)

        if self.debug:
            self.token_ids.scatter_(1, insertion_point[:, None], self.next_token_id)
            self.next_token_id += 1

        return insertion_point

    def remove(self, mask: torch.BoolTensor):
        expected_size = self._effective_size()
        assert mask.shape[0] == self.batch_size
        assert mask.shape[1] == expected_size
        assert len(mask.shape) == 2
        inv_mask = ~mask
        self.mask[:, :expected_size] &= inv_mask
        if self.debug:
            self.token_ids[:, :expected_size] *= inv_mask
            self.token_ids[:, :expected_size] += mask * (-1)

        if self.compact:
            # Compute load factor
            mask = self.mask[:, : self.max_padded_length]
            ratio = mask.sum(dim=1).max().item() / mask.shape[1]

        if self.compact and ratio < 0.9:
            # Find offset
            mask = self.mask[:, :expected_size]
            result = mask.min(dim=1)
            insertion_point = (
                result.indices * (~result.values) + mask.shape[1] * result.values
            )
            offset = insertion_point.min().item()
            if self.reduce_fragmentation and offset > 0:
                offset = 2 ** int(math.floor(math.log2(offset)))

            # Compact data structure
            indices = torch.argsort(~self.mask[:, offset:expected_size].long()) + offset
            self.mask[:, offset:expected_size] = self.mask.gather(1, indices)
            if self.debug:
                self.token_ids[:, offset:expected_size] = self.token_ids.gather(
                    1, indices
                )
            for i, (tensor, emb_dim) in enumerate(
                zip(self.tensors, self.embedding_dims)
            ):
                if isinstance(emb_dim, (tuple, list)):
                    indices_ = indices[:, None, :, None].expand(
                        -1, emb_dim[0], -1, emb_dim[1]
                    )
                    self.tensors[i][:, :, offset:expected_size] = tensor.gather(
                        2, indices_
                    )
                else:
                    indices_ = indices[:, :, None].expand(-1, -1, emb_dim)
                    self.tensors[i][:, offset:expected_size] = tensor.gather(
                        1, indices_
                    )

            # Find new max padded length
            mask_sum = torch.flip(self.mask[:, offset:expected_size].any(dim=0), (0,))
            result = mask_sum.max(dim=0)
            last_value = result.values.item()
            padded_length = mask_sum.shape[0] - result.indices.item() + offset
            if last_value:
                self.max_padded_length = padded_length
            else:
                self.max_padded_length = 0

    def values(self, tensor_ids=None):
        padded_length = self._effective_size()
        tensors = []
        for i, (tensor, emb_dim) in enumerate(zip(self.tensors, self.embedding_dims)):
            if tensor_ids is None or i in tensor_ids:
                tensors.append(tensor[..., :padded_length, :])
        return tensors, self.mask[:, :padded_length]

    def get_token_ids(self, compact=False):
        assert self.debug
        assert (self.token_ids[:, self.max_padded_length :] == -1).all()
        if compact:
            result = []
            for row in self.token_ids[:, : self._effective_size()]:
                ids = row[torch.where(row != -1)].sort().values
                result.append(ids)
            return result
        else:
            return self.token_ids[:, : self._effective_size()]

    def get_dense_mask(self) -> torch.BoolTensor:
        assert self.debug
        token_ids = self.token_ids[:, : self.max_padded_length]
        mask = torch.zeros(
            (self.batch_size, self.next_token_id + 1),
            dtype=torch.bool,
            device=token_ids.device,
        )

        # Index 0 is a dummy index to deal with gaps (token_id = -1)
        mask.scatter_(1, token_ids + 1, True)
        return mask[:, 1:]

    def get_dense_values(self):
        assert self.debug
        result = []
        for row_idx, row in enumerate(self.token_ids[:, : self._effective_size()]):
            ids = row.argsort()[(row == -1).sum() :]
            sub_result = []
            for tensor, emb_dim in zip(self.tensors, self.embedding_dims):
                if isinstance(emb_dim, (tuple, list)):
                    # Restore correct shape (number of heads)
                    tensor = tensor.view(self.batch_size, emb_dim[0], -1, emb_dim[1])
                sub_result.append(tensor[row_idx, ..., ids, :])
            result.append(sub_result)
        return result


class DynamicTensorReferenceDynamic:
    def __init__(
        self,
        batch_size,
        embedding_dims,
        capacity,
        resizable=True,
        reduce_fragmentation=False,
        dtype=torch.float32,
        device="cpu",
        debug=False,
    ):
        self.batch_size = batch_size
        self.embedding_dims = embedding_dims
        self.debug = debug
        self.capacity = capacity
        self.effective_length = 0
        self.reduce_fragmentation = reduce_fragmentation
        self.resizable = resizable

        self.tensors = []
        for embedding_dim in embedding_dims:
            if isinstance(embedding_dim, (list, tuple)):
                # Number of heads is specified
                assert len(embedding_dim) == 2
                self.tensors.append(
                    torch.zeros(
                        (batch_size, embedding_dim[0], capacity, embedding_dim[1]),
                        dtype=dtype,
                        device=device,
                    )
                )
            else:
                self.tensors.append(
                    torch.zeros(
                        (batch_size, capacity, embedding_dim),
                        dtype=dtype,
                        device=device,
                    )
                )

        self.mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)

        if self.debug:
            self.token_ids = torch.full(
                (batch_size, capacity), dtype=torch.long, device=device, fill_value=-1
            )
            self.next_token_id = 0

    def to(self, device=None, dtype=None):
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device=device, dtype=dtype)
        self.mask = self.mask.to(device=device)
        if self.debug:
            self.token_ids = self.token_ids.to(device=device)

    def _effective_size(self):
        if self.reduce_fragmentation and self.effective_length > 0:
            return 2 ** int(math.ceil(math.log2(self.effective_length)))
        else:
            return self.effective_length

    def _resize(self, new_capacity):
        for i, old_tensor in enumerate(self.tensors):
            if len(old_tensor.shape) == 4:
                new_tensor = torch.zeros(
                    (
                        old_tensor.shape[0],
                        old_tensor.shape[1],
                        new_capacity,
                        old_tensor.shape[3],
                    ),
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
            else:
                new_tensor = torch.zeros(
                    (old_tensor.shape[0], new_capacity, old_tensor.shape[2]),
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
            new_tensor[..., : self.capacity, :] = old_tensor[..., : self.capacity, :]
            self.tensors[i] = new_tensor

        new_mask = torch.zeros(
            (self.mask.shape[0], new_capacity),
            dtype=self.mask.dtype,
            device=self.mask.device,
        )
        new_mask[:, : self.capacity] = self.mask[:, : self.capacity]
        self.mask = new_mask

        if self.debug:
            new_token_ids = torch.full(
                (self.token_ids.shape[0], new_capacity),
                dtype=self.token_ids.dtype,
                device=self.token_ids.device,
                fill_value=-1,
            )
            new_token_ids[:, : self.capacity] = self.token_ids[:, : self.capacity]
            self.token_ids = new_token_ids

        self.capacity = new_capacity

    def append(self, tensors) -> torch.LongTensor:
        # Sanity check
        assert len(tensors) == len(self.embedding_dims)
        for tensor, embedding_dim in zip(tensors, self.embedding_dims):
            if isinstance(embedding_dim, (tuple, list)):
                # Number of heads is specified
                assert len(tensor.shape) == 3
                assert tensor.shape[0] == self.batch_size
                assert tensor.shape[1] == embedding_dim[0]
                assert tensor.shape[2] == embedding_dim[1]
            else:
                assert len(tensor.shape) == 2
                assert tensor.shape[0] == self.batch_size
                assert tensor.shape[1] == embedding_dim

        if self.effective_length == self.capacity:
            # Needs resizing
            if not self.resizable:
                raise RuntimeError(
                    "The pre-allocated buffer has been exhausted. "
                    "Increase the capacity or set resizable=True."
                )
            new_capacity = (self.capacity * 2) if self.capacity > 0 else 1
            self._resize(new_capacity)

        for i, tensor in enumerate(tensors):
            self.tensors[i][..., self.effective_length, :] = tensor
        self.mask[:, self.effective_length] = True

        if self.debug:
            self.token_ids[:, self.effective_length] = self.effective_length
            self.next_token_id += 1

        self.effective_length += 1
        return torch.full(
            (self.batch_size,),
            dtype=torch.long,
            device=self.mask.device,
            fill_value=self.effective_length - 1,
        )

    def remove(self, mask: torch.BoolTensor):
        expected_size = self._effective_size()
        assert mask.shape[0] == self.batch_size
        assert mask.shape[1] == expected_size
        assert len(mask.shape) == 2
        inv_mask = ~mask
        self.mask[:, :expected_size] &= inv_mask
        if self.debug:
            self.token_ids[:, :expected_size] *= inv_mask
            self.token_ids[:, :expected_size] += -1 * mask

    def values(self, tensor_ids=None):
        effective_length = self._effective_size()
        if tensor_ids is None:
            # Return all tensors
            tensors = [x[..., :effective_length, :] for x in self.tensors]
        else:
            tensors = [
                x[..., :effective_length, :]
                for i, x in enumerate(self.tensors)
                if i in tensor_ids
            ]
        return tensors, self.mask[:, :effective_length]

    def get_token_ids(self, compact=False):
        assert self.debug
        if compact:
            result = []
            for row in self.token_ids:
                ids = row[torch.where(row != -1)]
                result.append(ids)
            return result
        else:
            return self.token_ids

    def get_dense_mask(self) -> torch.BoolTensor:
        return self.mask[:, : self.effective_length]

    def get_dense_values(self):
        assert self.debug
        result = []
        for row_idx, row in enumerate(self.token_ids):
            (ids,) = torch.where(row != -1)
            sub_result = []
            for tensor in self.tensors:
                sub_result.append(tensor[row_idx, ..., ids, :])
            result.append(sub_result)
        return result
