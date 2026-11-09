from chug.common import parse_args

from pixparse.app.train import TrainCfg
from pixparse.app.eval import EvalCfg
from pixparse.task import (
    get_train_task_cfgs, get_eval_task_cfgs, TaskCrullerFinetuneDocVqaCfg, TaskCrullerPretrainCfg, TaskCrullerEvalDocVqaCfg
)


def test_train_cfg_parse(tmp_path):
    cfg = parse_args(
        TrainCfg,
        argv=[
            '--task', 'cruller_finetune_docvqa',
            '--task.model.name', 'cruller_naflex_clip_base', '--task.model.max-seq-len', '2048',
            '--task.opt.learning-rate', '3e-5', '--task.opt.betas', '0.9', '0.999',
            '--train-data.source', 'pixparse/docvqa-single-page-questions', '--train-data.split', 'train',
            '--train-data.format', 'hfds', '--eval-data.source', 'x', '--checkpoint-path', 'ckpt.pt',
        ],
        subgroups={'task': get_train_task_cfgs()},
    )
    assert isinstance(cfg.task, TaskCrullerFinetuneDocVqaCfg)
    assert cfg.task.model.max_seq_len == 2048 and cfg.task.opt.learning_rate == 3e-5
    assert cfg.task.opt.betas == (0.9, 0.999)
    assert cfg.train_data.format == 'hfds' and cfg.eval_data.source == 'x'
    # config file round trip w/ cli override
    p = tmp_path / 'cfg.yaml'
    cfg.save(p)
    import yaml
    d = yaml.safe_load(p.read_text())
    d['task']['_name'] = 'cruller_finetune_docvqa'
    p.write_text(yaml.safe_dump(d))
    cfg2 = parse_args(TrainCfg, argv=['--config', str(p), '--task.opt.learning-rate', '1e-4'],
                      subgroups={'task': get_train_task_cfgs()})
    assert isinstance(cfg2.task, TaskCrullerFinetuneDocVqaCfg)
    assert cfg2.task.opt.learning_rate == 1e-4 and cfg2.task.model.max_seq_len == 2048


def test_default_and_eval_cfg():
    cfg = parse_args(TrainCfg, argv=['--train-data.source', 'a.tar'], subgroups={'task': get_train_task_cfgs()},
                     subgroup_defaults={'task': 'cruller_pretrain'})
    assert isinstance(cfg.task, TaskCrullerPretrainCfg)
    ecfg = parse_args(EvalCfg, argv=['--checkpoint-path', 'c.pt'], subgroups={'task': get_eval_task_cfgs()},
                      subgroup_defaults={'task': 'cruller_eval_docvqa'})
    assert isinstance(ecfg.task, TaskCrullerEvalDocVqaCfg) and ecfg.checkpoint_path == 'c.pt'
