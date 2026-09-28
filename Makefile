# Pipeline stages. `make --dry-run all` lists every stage.
PY ?= python
PHASE ?= val
ALLOW ?=
FM_MODELS ?= chronos2 timesfm25 moirai2 tirex
CPU_MODELS ?= seasonal_naive operator

.PHONY: all data report forecast forecast-gpu tune calibrate calibrate-tune decide eval figures test freeze

all: data forecast calibrate decide eval figures

data:
	$(PY) -m tgc.data.build --download
	$(PY) -m tgc.data.report

report:
	$(PY) -m tgc.data.report

tune:
	$(PY) -m tgc.forecast.tune_lgbm

forecast:
	for m in $(CPU_MODELS); do $(PY) -m tgc.forecast.run --model $$m --phase $(PHASE) $(ALLOW); done
	for s in 0 1 2; do $(PY) -m tgc.forecast.run --model lgbm --seed $$s --phase $(PHASE) $(ALLOW); done

# Run on a GPU (Kaggle notebook): foundation models + deep baselines
forecast-gpu:
	for m in $(FM_MODELS); do $(PY) -m tgc.forecast.run --model $$m --phase $(PHASE) $(ALLOW); done
	for m in nhits patchtst; do for s in 0 1 2; do $(PY) -m tgc.forecast.run --model $$m --seed $$s --phase $(PHASE) $(ALLOW); done; done

calibrate-tune:
	$(PY) -m tgc.calibrate.run --tune

calibrate:
	$(PY) -m tgc.calibrate.run

decide:
	$(PY) -m tgc.decide.run --phase $(PHASE) $(ALLOW)

eval:
	$(PY) -m tgc.evaluate.run --phase $(PHASE) $(ALLOW)

figures:
	$(PY) -m tgc.plots.figures --phase $(PHASE) $(ALLOW)
	$(PY) -m tgc.plots.tables --phase $(PHASE)

test:
	$(PY) -m pytest -q

# Freeze validation-tuned settings before the single test run (Section 12.5)
freeze:
	git add configs && git commit -m "Freeze validation-tuned configs" && git tag v-frozen
