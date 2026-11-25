# Cape Verdean Creole ↔ Portuguese Translation

Machine translation between Kabuverdianu (Cape Verdean Creole) varieties and European Portuguese, built by fine-tuning multilingual seq2seq models.

## Language

**Variety**:
A regional form of Kabuverdianu treated by the model as its own language (e.g. Santiago, São Vicente).
_Avoid_: dialect, variant, language (for Kabuverdianu sub-forms)

**Variety token**:
The dedicated special token that identifies one Variety to the model, one per Variety (`kea_st_XX` for Santiago, `kea_sv_XX` for São Vicente).
_Avoid_: language code, `kea_Latn` (NLLB's generic Kabuverdianu token, never used to mark a Variety)

**Seed token**:
An existing token whose embedding is copied into a newly added Variety token as its starting value; the Variety token keeps its own id.
_Avoid_: alias, mapping

**Corpus**:
A parallel Kabuverdianu–Portuguese dataset bound to exactly one Variety.
_Avoid_: dataset (when the Variety binding matters)

**Bidirectional run**:
A training run that learns both Variety → Portuguese and Portuguese → Variety from the same Corpus.
