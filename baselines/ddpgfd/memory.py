import numpy as np


class DemoRingBuffer(object):
    def __init__(self, maxlen, data, nb_min_demo,shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.nb_min_demo = nb_min_demo
        self.length = nb_min_demo
        self.data = np.zeros((maxlen,) + shape).astype(dtype)
        for i in range(len(data)):
            self.data[i] =  data[i]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return np.asarray(self.data[(self.start + idx) % self.maxlen])

    def get_batch(self, idxs):
        self.data_array=np.asarray(self.data)
        return self.data_array[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the (nb-min-demo + 1)-th element
            # and replace with the new one
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length + self.nb_min_demo - 1) % (self.maxlen - self.nb_min_demo)] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape, nb_min_demo, demonstrations):
        self.limit = limit
        # Minimum number of demonstration to be included in the buffer
        assert( nb_min_demo > 0 )
        self.nb_min_demo = nb_min_demo


        self.observations0 = DemoRingBuffer(limit,demonstrations.obs0, nb_min_demo, shape=observation_shape)
        self.actions = DemoRingBuffer(limit, demonstrations.acts,nb_min_demo,shape=action_shape)
        self.rewards = DemoRingBuffer(limit, demonstrations.rewards,nb_min_demo ,shape=(1,))
        self.terminals1 = DemoRingBuffer(limit, demonstrations.terms, nb_min_demo,shape=(1,))
        self.observations1 = DemoRingBuffer(limit, demonstrations.obs1,nb_min_demo,shape=observation_shape)
        self.observationsn = DemoRingBuffer(limit, demonstrations.obsn,nb_min_demo,shape=observation_shape)
        self.terminalsn = DemoRingBuffer(limit, demonstrations.termsn,nb_min_demo,shape=(1,))
        self.rewardsn = DemoRingBuffer(limit, demonstrations.rewardsn,nb_min_demo,shape=(1,))

        #import IPython
        #IPython.embed()

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        self.batch_idxs = batch_idxs

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)
        rewardn_batch = self.rewardsn.get_batch(batch_idxs)
        terminaln_batch = self.terminalsn.get_batch(batch_idxs)
        obsn_batch = self.observationsn.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'obsn': array_min2d(obsn_batch),
            'rewards': array_min2d(reward_batch),
            'rewardsn': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'terminalsn': array_min2d(terminaln_batch),
        }
        return result

    def sample_with_priorization(self, batch_size, priority):
        # Draw such that we always have a proceeding element.

        print(priority)
        priority[:,0] = priority[:,0]/  np.sum(priority[:,0])
        #priority= priority/  np.sum(priority)
        print('in memory', np.sum(priority[:,0]))

        self.batch_idxs = np.random.choice(self.nb_entries, size=batch_size, p=priority[:,0])
        #self.batch_idxs = np.random.choice(self.nb_entries, size=batch_size, p=priority)

        obs0_batch = self.observations0.get_batch(self.batch_idxs)
        obs1_batch = self.observations1.get_batch(self.batch_idxs)
        action_batch = self.actions.get_batch(self.batch_idxs)
        reward_batch = self.rewards.get_batch(self.batch_idxs)
        terminal1_batch = self.terminals1.get_batch(self.batch_idxs)
        rewardn_batch = self.rewardsn.get_batch(self.batch_idxs)
        terminaln_batch = self.terminalsn.get_batch(self.batch_idxs)
        obsn_batch = self.observationsn.get_batch(self.batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'obsn': array_min2d(obsn_batch),
            'rewards': array_min2d(reward_batch),
            'rewardsn': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'terminalsn': array_min2d(terminaln_batch),
        }
        return result

    def append(self, obs0, obs1, obsn, action, reward, terminal1, terminalsn, rewardn, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)
        self.observationsn.append(obsn)
        self.terminalsn.append(terminalsn)
        self.rewardsn.append(rewardn)

    @property
    def nb_entries(self):
        #return len(self.observations0)
        return len(self.observations0)
