import numpy as np


class DemoRingBuffer(object):
    def __init__(self, maxlen, data, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.nb_min_demo = nb_min_demo
        self.length = 0
        self.data = data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return  self.data[(self.start + idxs) % self.maxlen]

    def get_batch(self, idxs):

        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the (nb-min-demo + 1)-th element
            # since we need to preserve at least nb-min-demo elements
            # from the original demonstrations
            self.data[i for i in arange(self.nb_min_demo, self.lenght - 2)] = self.data[(i + 1  for i in arange(self.nb_min_demo, self.lenght - 2))]
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, nb_min_demo, demonstrations, alpha):
        self.limit = limit
        # Minimum number of demonstration to be included in the buffer
        assert( nb_min_demo > 0 )
        self.nb_min_demo = nb_min_demo
        assert( alpha > 0 )
        self.alpha = alpha

        assert(observation_shape == demonstrations.obs0)
        self.observations0 = DemoRingBuffer(limit,demonstrations.obs0)
        assert(action_shape == demonstrations.acts)
        self.actions = DemoRingBuffer(limit, demonstrations.acts)
        self.rewards = DemoRingBuffer(limit, demonstrations.rewards)
        self.terminals1 = DemoRingBuffer(limit, demonstrations.terminals)
        #assert(observation_shape == demonstrations.obs1)
        #self.observations1 = DemoRingBuffer(limit, demonstrations.obs1)  ## maybe demonstration here are not necessary

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def sample_with_priorization(self, batch_size, priority):
        # Draw such that we always have a proceeding element.
        priority_alpha = priority ** self.alpha
        priority_alpha = priority_alpha / np.sum(priority_alpha)
        batch_idxs = np.random.choice(self.nb_entries - 2, size=batch_size, p=priority_alpha))

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)
