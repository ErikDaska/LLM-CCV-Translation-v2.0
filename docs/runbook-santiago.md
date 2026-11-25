# Runbook: treinar e avaliar a variedade de Santiago (`kea_st_XX`)

Guia passo a passo para treinar o modelo São Vicente + Santiago no datacenter (via SSH) e verificar se São Vicente regrediu. A decisão por trás deste fluxo está em [ADR 0001](adr/0001-sequential-variety-fine-tuning.md).

**Resumo:** continua-se o treino do modelo publicado `Iscte-Sintra/nllb600M-kea-pt-bidirectional` (SV) apenas com o corpus de Santiago (`Iscte-Sintra/Korpora_paralela_st_clean`). O novo token `kea_st_XX` começa como cópia de `kea_sv_XX`.

---

## 1. Preparar o datacenter

```bash
ssh <datacenter>
cd <caminho>/LLM-CCV-Translation-v2.0
git pull
source venv/bin/activate
pip install -r requirements.txt
mkdir -p logs
```

O `.env` tem de ter o token do Hugging Face: o modelo-pai e o dataset são privados.

## 2. Lançar o treino (sobrevive ao fecho da sessão SSH)

```bash
nohup python src/train.py --config-name manual_nllb_bidirectional_st \
  > logs/kea_st.log 2>&1 &
echo $! > logs/kea_st.pid
```

| Parte | O que faz |
|---|---|
| `nohup` | ignora o sinal de "hang up": fechar o SSH não mata o treino |
| `> logs/kea_st.log 2>&1` | todo o output e erros vão para o ficheiro de log |
| `&` | corre em segundo plano e devolve o terminal |
| `echo $! > logs/kea_st.pid` | guarda o PID para consultar ou parar mais tarde |

Antes de sair, confirmar que arrancou bem:

```bash
tail -f logs/kea_st.log     # Ctrl+c só pára de ver o log, não o treino
```

Procurar a linha `Initialized kea_st_XX (id …) from kea_sv_XX` e os primeiros passos de treino. Depois disso, pode fechar o SSH.

> Se mesmo assim o treino morrer ao fechar o SSH, lançar com `setsid nohup python ... &`.

### O que o treino faz

1. Compõe a config Hydra: `manual_nllb_bidirectional_st` + `model/nllb` + `data/kea_st_pt` + `training_arguments/manual_nllb`.
2. Descarrega o corpus de Santiago e remove as linhas `en-us` (ficam 17 883 de treino).
3. Descarrega o modelo SV publicado.
4. Regista `kea_st_XX` e copia o embedding de `kea_sv_XX` como ponto de partida.
5. Treina nas duas direções (kea_st→pt e pt→kea_st), até 10 épocas, LR 2e-5.
6. Avalia no fim de cada época e guarda o melhor checkpoint por `eval_chrf_mean`. Early stopping com paciência 3.
7. Grava o melhor modelo em `<run_dir>/model`. **Não** publica no Hub (`push_to_hub: false`).

### Acompanhar e controlar

```bash
tail -f logs/kea_st.log               # ver progresso
ps -p $(cat logs/kea_st.pid)          # ainda está a correr?
kill $(cat logs/kea_st.pid)           # parar o treino
```

## 3. Avaliar depois do treino

### 3.1 Encontrar a pasta do run

```bash
grep "Run " logs/kea_st.log | head -1
ls /home/criolo/storage/models_outputs/translation_models/nllb600m-kea-sv-st-pt-bidir/
```

O melhor modelo está em `<run_dir>/model`. Não usar as pastas `checkpoints/checkpoint-N`.

### 3.2 Apontar as configs de avaliação para o modelo novo

```bash
RUN=/home/criolo/storage/models_outputs/translation_models/nllb600m-kea-sv-st-pt-bidir/<pasta-do-run>
ls $RUN/model        # deve ter config.json e model.safetensors
sed -i "s#^  model_id: /home/criolo/.*#  model_id: $RUN/model#" src/configs/evaluation/nllb_sv_st_*.yaml
grep model_id src/configs/evaluation/nllb_sv_*.yaml
```

Resultado esperado: 4 ficheiros `nllb_sv_st_*` apontam para `$RUN/model` e 2 ficheiros `nllb_sv_parent_*` continuam em `Iscte-Sintra/nllb600M-kea-pt-bidirectional`.

### 3.3 Correr as 6 avaliações em segundo plano

```bash
nohup bash -c 'for c in src/configs/evaluation/nllb_sv_*.yaml; do
  echo "=== $c"; python src/evaluate_models.py --config "$c"
done' > logs/kea_st_eval.log 2>&1 &
tail -f logs/kea_st_eval.log
```

| Config | Modelo | Direção | Para quê |
|---|---|---|---|
| `nllb_sv_st_st_to_pt` | novo | kea_st → pt | qualidade de Santiago |
| `nllb_sv_st_pt_to_st` | novo | pt → kea_st | qualidade de Santiago |
| `nllb_sv_st_sv_to_pt` | novo | kea_sv → pt | São Vicente regrediu? |
| `nllb_sv_st_pt_to_sv` | novo | pt → kea_sv | São Vicente regrediu? |
| `nllb_sv_parent_sv_to_pt` | pai (Hub) | kea_sv → pt | baseline SV |
| `nllb_sv_parent_pt_to_sv` | pai (Hub) | pt → kea_sv | baseline SV |

Todos os resultados ficam no W&B, no grupo `kea_sv_st_regression`.

## 4. Decidir

Comparar o chrF:

- `nllb_sv_st_sv_to_pt` vs `nllb_sv_parent_sv_to_pt`
- `nllb_sv_st_pt_to_sv` vs `nllb_sv_parent_pt_to_sv`

| Resultado | Ação |
|---|---|
| SV igual ou melhor que o pai | pôr `push_to_hub: true` em `manual_nllb_bidirectional_st.yaml` e publicar como `Iscte-Sintra/nllb600M-kea-sv-st-pt-bidirectional` |
| SV caiu claramente | retreinar com LR menor, por exemplo `python src/train.py --config-name manual_nllb_bidirectional_st training_arguments.learning_rate=1e-5`; se continuar a cair, rever a opção de *replay* (ADR 0001) |

Nunca publicar por cima de `Iscte-Sintra/nllb600M-kea-pt-bidirectional`.
