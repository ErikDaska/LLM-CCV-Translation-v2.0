"""Load explicit experiment sections and validate settings before resource loading."""
from pathlib import Path
import yaml


SECTIONS = {
    'model': {'model', 'src_lang', 'tgt_lang'},
    'data': {'dataset', 'max_length'},
    'tracking': {'project', 'group_name', 'run_name', 'output_base_dir'},
    'hub': {'push_to_hub', 'hub_repo_id', 'hub_private'},
    'workflow': {'mode'},
    'hpo': {'objective', 'direction', 'trials', 'search_space'},
    'training': None,
}


def load_configuration(path):
    """Return the nested configuration and a compatibility view for application fields.

    Trainer settings remain nested and are forwarded directly to TrainingArguments.
    Unknown Trainer arguments are rejected by its constructor before model loading.
    """
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not isinstance(config.get('workflow'), dict):
        raise ValueError('Configuration requires a workflow mapping with mode: manual or hpo.')
    mode = config['workflow'].get('mode')
    if mode not in ('manual', 'hpo'):
        raise ValueError('workflow.mode must be manual or hpo.')
    sections = dict(SECTIONS)
    if mode == 'manual':
        # Este pop está aqui a fazer o quê? Partimos do pressuposto que o manual não terá informações sobre o HPO
        sections.pop('hpo')
        sections['workflow'] = {'mode', 'early_stopping_patience'}
    if set(config) != set(sections):
        raise ValueError(f'{mode} configuration must contain exactly these sections: {list(sections)}')

    application = {}
    for section, keys in sections.items():
        values = config[section]
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be a mapping.')
        if keys is not None and set(values) != keys:
            raise ValueError(f'{section} must contain exactly these keys: {sorted(keys)}')
        if section not in ('training', 'hpo'):
            application.update(values)
    training = config['training']

    if {'output_dir', 'run_name', 'push_to_hub', 'hub_token'} & training.keys():
        raise ValueError('Run paths and Hub publication are managed by the application.')
    required = {'predict_with_generate', 'report_to', 'gradient_checkpointing',
                'warmup_steps', 'warmup_ratio', 'fp16', 'bf16', 'load_best_model_at_end'}
    search_space = config['hpo']['search_space'] if mode == 'hpo' else {}
    if not isinstance(search_space, dict) or (mode == 'hpo' and not search_space):
        raise ValueError('hpo.search_space must be a non-empty mapping.')
    duplicates = training.keys() & search_space.keys()
    if duplicates:
        raise ValueError(f'Parameters cannot be both fixed and searched: {sorted(duplicates)}')
    required -= search_space.keys()
    if not required <= training.keys():
        raise ValueError(f'Missing training settings: {sorted(required - training.keys())}')
    for key in ('learning_rate', 'weight_decay', 'warmup_ratio', 'label_smoothing_factor'):
        if key in training:
            training[key] = float(training[key])

    if not training['predict_with_generate']:
        raise ValueError('Translation metrics require predict_with_generate: true.')

    if training['report_to'] != ['wandb']:
        raise ValueError('This workflow currently requires report_to: [wandb].')
    if training['fp16'] and training['bf16']:
        raise ValueError('Enable only one of fp16 and bf16.')
    if training.get('warmup_steps', 0) > 0 and training.get('warmup_ratio', 0) > 0:
        raise ValueError('Choose warmup_steps or warmup_ratio, not both.')
    if mode == 'manual':
        if application['early_stopping_patience'] and not training['load_best_model_at_end']:
            raise ValueError('Early stopping requires load_best_model_at_end.')
        return config, application

    hpo = config['hpo']
    if hpo['objective'] not in {'eval_chrf', 'eval_SacreBleu', 'eval_meteor', 'eval_ter', 'eval_loss'}:
        raise ValueError('Unknown HPO objective.')
    expected = 'minimize' if hpo['objective'] in {'eval_ter', 'eval_loss'} else 'maximize'
    if hpo['direction'] != expected:
        raise ValueError(f"{hpo['objective']} requires direction: {expected}")
    if not isinstance(hpo['trials'], int) or hpo['trials'] < 1:
        raise ValueError('hpo.trials must be a positive integer.')
    for name, spec in hpo['search_space'].items():
        # Limit search to parameters supported by this workflow. Runtime controls
        # (precision, evaluation, paths, etc.) remain fixed and validated above.
        if name not in {'learning_rate', 'per_device_train_batch_size', 'num_train_epochs',
                        'label_smoothing_factor', 'warmup_ratio', 'weight_decay'}:
            raise ValueError(f'Unsupported HPO parameter: {name}')
        if not isinstance(spec, dict):
            raise ValueError(f'Search specification must be a mapping: {name}')
        kind = spec.get('type')
        if kind == 'categorical':
            if set(spec) != {'type', 'choices'} or not spec['choices']:
                raise ValueError(f'Invalid categorical search specification: {name}')
        elif kind in ('float', 'int'):
            if not {'type', 'low', 'high'} <= spec.keys() or set(spec) - {'type', 'low', 'high', 'log'}:
                raise ValueError(f'Invalid numeric search specification: {name}')
            if spec['low'] > spec['high'] or (spec.get('log') and spec['low'] <= 0):
                raise ValueError(f'Invalid search bounds: {name}')
        else:
            raise ValueError(f'Unknown search type for {name}: {kind}')
    if 'warmup_ratio' in hpo['search_space'] and training['warmup_steps'] > 0:
        raise ValueError('Set warmup_steps to zero when searching warmup_ratio.')
    return config, application
