"""Resumable sample order, independent random streams, and FP32 EMA."""

import copy

import torch

from flow_matching.timesteps import sample_timesteps


class SampleOrder:
    """Infinite shuffled epochs; checkpoint only the committed epoch/offset."""

    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("Training data must be nonempty.")
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.offset = 0
        self._cached_epoch = None
        self._permutation = None

    def take(self, count):
        indices = []
        while count:
            if self._cached_epoch != self.epoch:
                generator = torch.Generator()
                generator.manual_seed((self.seed + self.epoch) % 2**63)
                self._permutation = torch.randperm(self.size, generator=generator)
                self._cached_epoch = self.epoch
            length = min(count, self.size - self.offset)
            selection = self._permutation[self.offset:self.offset + length]
            indices.extend(selection.tolist())
            self.offset += length
            count -= length
            if self.offset == self.size:
                self.epoch += 1
                self.offset = 0
        return indices

    def state_dict(self):
        return {"size": self.size, "seed": self.seed, "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state):
        if state["size"] != self.size or state["seed"] != self.seed:
            raise ValueError("Sample-order dataset/seed differs from checkpoint.")
        self.epoch = state["epoch"]
        self.offset = state["offset"]
        self._cached_epoch = None


class MicrobatchSampler:
    """Prefetch with a private cursor; only the trainer commits sample order."""

    def __init__(self, order, config, rank):
        self.order = SampleOrder(order.size, order.seed)
        self.order.load_state_dict(order.state_dict())
        self.config = config
        self.rank = rank

    def __iter__(self):
        config = self.config
        while True:
            indices = self.order.take(config.global_batch)
            for micro in range(config.accumulation):
                start = (micro * config.processes + self.rank) * config.microbatch
                yield indices[start:start + config.microbatch]


class TrainingState:
    """All explicit streams advance identically on every rank and across K."""

    def __init__(self, seed, data_size):
        self.step = 0
        self.order = SampleOrder(data_size, seed + 1)
        self.generators = {}
        for offset, name in enumerate(("augmentation", "exposure", "timestep"), start=2):
            generator = torch.Generator()
            generator.manual_seed(seed + offset)
            self.generators[name] = generator

    def draw_conditions(self, size, data_config, timesteps):
        augmentation = torch.rand(size, generator=self.generators["augmentation"])
        flips = augmentation < data_config.train_flip_probability
        uniforms = torch.rand((size, 1), generator=self.generators["exposure"])
        exposure_range = data_config.exposure_max - data_config.exposure_min
        exposure = data_config.exposure_min + uniforms * exposure_range
        times = sample_timesteps(size, timesteps, self.generators["timestep"])
        return flips, exposure, times

    def state_dict(self):
        generators = {}
        for name, generator in self.generators.items():
            generators[name] = generator.get_state()
        state = {"step": self.step, "order": self.order.state_dict(), "generators": generators}
        return state

    def load_state_dict(self, state):
        self.step = state["step"]
        self.order.load_state_dict(state["order"])
        for name, values in state["generators"].items():
            self.generators[name].set_state(values.cpu())


def make_ema(model):
    averaged = copy.deepcopy(model)
    averaged.requires_grad_(False)
    averaged.eval()
    return averaged


@torch.no_grad()
def update_ema(averaged, model, decay):
    for target, source in zip(averaged.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach().float(), 1 - decay)
    for target, source in zip(averaged.buffers(), model.buffers(), strict=True):
        target.copy_(source)
