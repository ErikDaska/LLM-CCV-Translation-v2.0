# LLM-CCV-Translation-v2.0

Treino e avaliação de modelos de tradução entre o Crioulo de São Vicente (KEA)
e o Português (PT), com mBART e NLLB, usando Hugging Face Transformers, W&B
para registo de experiências e Hydra para gestão de configuração.

## Requisitos

- Python 3.12 (o `venv/` do projeto já está assim configurado)
- GPU NVIDIA para treino (os configs usam `bf16: true`; usa `fp16`/`bf16: false`
  se o dispositivo não suportar BF16)
- Contas: Hugging Face (token) e Weights & Biases

## Instalação

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuração (`.env`)

Crie um ficheiro `.env` na raiz do projeto (não é versionado):

| Variável | Obrigatória | Descrição |
|---|---|---|
| `WANDB_API_KEY` | Sim | Chave do Weights & Biases |
| `HF_TOKEN` | Sim | Token de acesso ao Hugging Face |
| `HF_HOME` | Não | Diretório de cache dos modelos/datasets |
| `WANDB_DIR` | Não | Diretório de trabalho do W&B |
| `TMPDIR` | Não | Diretório temporário |
| `MONGODB_KEY` | Só notebooks | URI do MongoDB usada na estatística de corpus |

## Estrutura de configuração (Hydra)

Os configs de treino vivem em `src/configs/training/`. Um **root** define a
identidade do experimento; os **groups** contêm partes reutilizáveis:

```
src/configs/training/
├── smoke_manual.yaml                # root: teste de integração (config por omissão)
├── manual_mbart51_sv.yaml           # root: run manual mBART-51 → Hub
├── manual_nllb_sv.yaml              # root: baseline manual NLLB (unidirecional)
├── manual_nllb_bidirectional.yaml   # root: NLLB kea<->pt no kea-sv-pt-v2
├── model/                           # group → chave "model"
│   ├── mbart.yaml                   #   facebook/mbart-large-50
│   ├── mbart51.yaml                 #   Iscte-Sintra/mbart-51-stable
│   └── nllb.yaml                    #   facebook/nllb-200-distilled-600M
├── data/                            # group → chave "data"
│   ├── sv.yaml                      #   dataset SV + colunas origem/alvo
│   └── kea_sv_pt.yaml               #   kea-sv-pt-v2 + colunas + bidirecional
└── training_arguments/              # group → chave "training_arguments"
    ├── smoke.yaml
    ├── manual_mbart51.yaml
    └── manual_nllb.yaml
```

Cada root tem uma lista `defaults` que escolhe os componentes e um bloco
`_self_` com `tracking`, `hub` e `early_stopping_patience`. Exemplo
(`manual_nllb_sv.yaml`):

```yaml
defaults:
  - model: nllb
  - data: sv
  - training_arguments: manual_nllb
  - _self_
early_stopping_patience: 3
tracking: { ... }
hub: { ... }
```

Os ficheiros de avaliação (`src/configs/evaluation/`) são YAML simples e
independentes do Hydra.

### Grupo `data`

Para além de `dataset` e `max_length`, o grupo `data` aceita:

| Chave | Padrão | Descrição |
|---|---|---|
| `source_column` | `original_sentence` | Coluna do texto de origem |
| `target_column` | `translation` | Coluna do texto de destino (referência de avaliação) |
| `bidirectional` | `false` | Treina as duas direções (origem↔destino) no mesmo run |
| `multi_reference_validation` | `false` | Agrupa fontes repetidas na validação (N referências) |
| `filters` | `{}` | Igualdades coluna=valor aplicadas a todos os splits (ex.: descartar linhas `en-us`) |

## Treino manual

Sempre a partir da raiz do projeto (com o `venv` ativo):

```bash
# Config por omissão (smoke: 2 passos, verificação de integração)
python src/train.py

# Um experimento concreto
python src/train.py --config-name manual_nllb_sv
python src/train.py --config-name manual_mbart51_sv
python src/train.py --config-name manual_nllb_bidirectional

# Dry-run: imprime a config composta e termina (não treina nem toca em GPU/W&B)
python src/train.py --cfg job --config-name manual_nllb_sv
```

### Treino bidirecional (kea ↔ pt)

`manual_nllb_bidirectional` usa o dataset `Iscte-Sintra/kea-sv-pt-v2`
(colunas `sentence` = crioulo, `translation_text` = português; uma linha
`en-us` descartada por `data.filters`):

- **Treino**: cada par é tokenizado nas duas direções
  (`kea→pt` e `pt→kea`) e concatenados; o token de língua da direção contrária
  é registado pelo próprio script (`_prepare_languages`, sem alterações).
- **Seleção de checkpoint**: só na direção `kea→pt` (`eval_chrf`), igual ao
  baseline unidirecional.
- **Avaliação final**: além da direção principal, o mesmo melhor checkpoint é
  avaliado em `pt→kea` (troca de `forced_bos_token_id` e referências) e
  gravado em `metrics.json` sob `validation_bidirectional` (prefixo W&B
  `eval_bidirectional_*`).

### Label smoothing

`training_arguments.label_smoothing_factor > 0` é suportado em m2m-100/MBART/
NLLB: o script anexa `prepare_decoder_input_ids_from_labels` ao modelo
(`TrainingTranslationScript._enable_label_smoothing_support`), o ponto de
extensão que o `DataCollatorForSeq2Seq` usa para reconstruir as entradas do
decoder quando o Trainer remove os `labels` do batch. Sem este hook, o
forward falharia com `ValueError: You cannot specify both decoder_input_ids
and decoder_inputs_embeds`.

### Overrides de hiperparâmetros (sem reescrever ficheiros)

Qualquer chave aceita `chave=valor` na linha de comandos, com sintaxe de ponto:

```bash
python src/train.py --config-name manual_nllb_sv \
    training_arguments.learning_rate=3e-5 \
    training_arguments.weight_decay=0.01 \
    tracking.run_name=nllb-lr3e-5
```

### Sweeps automáticos

O modo `-m` (multirun) executa o produto cartesiano dos valores listados. Cada
combinação gera um run independente com o seu ficheiro de config composto —
não é preciso escrever nenhum YAML:

```bash
python src/train.py -m --config-name manual_nllb_sv \
    training_arguments.learning_rate=1e-5,2e-5,3e-5 \
    training_arguments.weight_decay=0.01,0.1
```

## Avaliação de modelos

```bash
# Baseline NLLB (config por omissão)
python src/evaluate_models.py

# Um checkpoint local ou do Hub
python src/evaluate_models.py --config src/configs/evaluation/mbart_local.yaml
python src/evaluate_models.py --config src/configs/evaluation/nllb_local.yaml
python src/evaluate_models.py --config src/configs/evaluation/m2m100_local.yaml
```

Os modelos locais referenciam um checkpoint pelo caminho; `model.src_lang`/`tgt_lang`
podem ser `auto` para lerem os códigos do `config.json` do run de treino.

## Onde mudar o quê

| O que mudar | Onde |
|---|---|
| Hiperparâmetros fixos partilhados | `src/configs/training/training_arguments/*.yaml` |
| Modelo / línguas | `src/configs/training/model/*.yaml` |
| Dataset / colunas / `max_length` / bidirecional | `src/configs/training/data/*.yaml` |
| Valores pontuais de um run | Override na linha de comandos (`training_arguments.x=...`) |
| Experimento novo | Copiar um root e ajustar `defaults` + `tracking`/`hub` |
| Configuração de avaliação | `src/configs/evaluation/*.yaml` |

## Saídas e registos

- **Run dir local**: `tracking.output_base_dir/<run_id>/` com `status.json`,
  `config.json`, `training_args.json`, `metrics.json`, `history.json`,
  `selection.json`, `wandb.json` e, no fim, `model/`.
- **Hydra**: em `outputs/<data>/<hora>/.hydra/config.yaml` fica a config
  composta exata de cada run (em sweeps: `multirun/<data>/<hora>/<n>/...`).
  Os diretórios `outputs/` e `multirun/` são ignorados pelo git.
- **W&B**: métricas e gráficos no projeto definido em `tracking.project`.

## Comandos rápidos

| Tarefa | Comando |
|---|---|
| Smoke test | `python src/train.py` |
| Treino manual | `python src/train.py --config-name manual_nllb_sv` |
| Treino bidirecional kea↔pt | `python src/train.py --config-name manual_nllb_bidirectional` |
| Override pontual | `python src/train.py --config-name X training_arguments.learning_rate=3e-5` |
| Sweep | `python src/train.py -m --config-name X a.b=1,2` |
| Ver config composta | `python src/train.py --cfg job --config-name X` |
| Avaliar modelo | `python src/evaluate_models.py --config src/configs/evaluation/nllb_local.yaml` |
| Limpar corpus | `python src/prepare_data.py` (parâmetros no topo do ficheiro) |

## Nota de migração

Os comandos antigos com caminho de ficheiro deixaram de existir no treino:

```bash
# Antes
python src/train.py --config src/configs/smoke/manual.yaml
# Agora
python src/train.py --config-name smoke_manual
```
