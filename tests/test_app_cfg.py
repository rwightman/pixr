import yaml

from pixr.app.eval import EvalCfg
from pixr.app.train import TrainCfg
from pixr.task import DocVqaTaskCfg, ReadTaskCfg, get_task_cfgs
from pixr.utils.cli import parse_cfg


def _train(argv):
    return parse_cfg(TrainCfg, argv=argv, task_cfgs=get_task_cfgs(), default_task='read')


def test_train_cfg_dashes_and_underscores():
    cfg = _train([
        '--task', 'docvqa',
        '--task.model.name', 'cruller_naflex_siglip2_base', '--task.model.max-seq-len', '2048',
        '--trainer.opt.learning-rate', '3e-5', '--trainer.opt.betas', '[0.9, 0.999]',
        '--trainer.opt.min_learning_rate=1e-6',
        '--train-data.source', 'pixparse/docvqa-single-page-questions', '--train-data.split', 'train',
        '--train-data.format', 'hfds', '--train-data.naflex-seq-lens', '[1024, 2048]',
        '--eval-data.source', 'x', '--checkpoint-path', 'ckpt.pt',
    ])
    assert cfg.task_name == 'docvqa' and isinstance(cfg.task, DocVqaTaskCfg)
    assert cfg.task.model.max_seq_len == 2048 and cfg.trainer.opt.learning_rate == 3e-5
    assert cfg.trainer.opt.betas == (0.9, 0.999) and cfg.trainer.opt.min_learning_rate == 1e-6
    assert cfg.train_data.format == 'hfds' and cfg.train_data.naflex_seq_lens == [1024, 2048]
    assert cfg.eval_data.source == 'x' and cfg.checkpoint_path == 'ckpt.pt'


def test_config_file_roundtrip(tmp_path):
    cfg = _train(['--task', 'docvqa', '--task.text-max-length', '96', '--trainer.num-intervals', '7'])
    p = tmp_path / 'cfg.yaml'
    cfg.save(p)
    assert yaml.safe_load(p.read_text())['task_name'] == 'docvqa'
    cfg2 = _train(['--config', str(p), '--trainer.opt.learning-rate', '1e-4'])  # task from the config file
    assert isinstance(cfg2.task, DocVqaTaskCfg) and cfg2.task.text_max_length == 96
    assert cfg2.trainer.num_intervals == 7 and cfg2.trainer.opt.learning_rate == 1e-4


def test_default_and_eval_cfg():
    cfg = _train(['--train-data.source', 'a.tar'])
    assert cfg.task_name == 'read' and isinstance(cfg.task, ReadTaskCfg)
    ecfg = parse_cfg(EvalCfg, argv=['--checkpoint-path', 'c.pt'], task_cfgs=get_task_cfgs(), default_task='docvqa')
    assert isinstance(ecfg.task, DocVqaTaskCfg) and ecfg.checkpoint_path == 'c.pt'


def test_image_aug_cfg(tmp_path):
    from chugr import DocAugCfg
    from pixr.data import ImagePreprocessorFactory
    cfg = _train([
        '--task.image-aug.bitmap-prob', '0.2', '--task.image-aug.shear-range-x', '[0, 5]',
        '--task.image-aug.prob_scale=0.5',
    ])
    aug = cfg.task.image_aug
    assert isinstance(aug, DocAugCfg) and aug.bitmap_prob == 0.2 and aug.shear_range_x == (0., 5.)
    assert aug.prob('bitmap_prob') == 0.1 and aug.elastic_prob == DocAugCfg().elastic_prob
    p = tmp_path / 'cfg.yaml'
    cfg.save(p)
    assert _train(['--config', str(p)]).task.image_aug == aug

    info = dict(naflex=True, image_size=None, image_fmt='L', mean=(0.5,), std=(0.5,), patch_size=16, max_seq_len=1024)
    t = ImagePreprocessorFactory(info, True, aug_cfg=DocAugCfg(prob_scale=0.))(512)
    assert not [op for op in t.transforms if type(op).__name__ == 'RandomApply']
