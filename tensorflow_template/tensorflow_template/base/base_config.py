import os
from huaytools import Bunch


class Config(Bunch):
    def __init__(self, name="", **kwargs):
        super(Config, self).__init__()

        self.name = name

        self.out_dir = './out_' + self.name
        os.makedirs(self.out_dir, exist_ok=True)

        self.ckpt_dir = os.path.join(self.out_dir, 'ckpt_' + self.name)
        self.eval_dir = os.path.join(self.out_dir, 'eval_' + self.name)
        self.summary_dir = os.path.join(self.out_dir, 'summary_' + self.name)  # no use

        self.n_feature = None
        self.n_class = None

        self.n_batch = 32
        self.n_epoch = 5
        self.n_step = None  # if use `tf.data.Dataset`, the step is decided by `n_epoch` and `n_batch`

        self.learning_rate = 0.001
        # self.ckpt_path = os.path.join(self.ckpt_dir, self.name)

        self.sess_config = None  # ref: `tf.ConfigProto`
        for k, v in kwargs.items():
            self[k] = v


if __name__ == '__main__':
    config = Config('', n_batch=11, aaa="AAA")
    config.ttttt = 'ttttt'

    print(config.n_batch)  # 11
    print(config.aaa)  # AAA

    print(config.ttttt)  # ttttt
    print(config['ttttt'])  # ttttt
