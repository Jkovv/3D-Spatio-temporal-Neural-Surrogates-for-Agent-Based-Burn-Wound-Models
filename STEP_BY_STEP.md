# STEP BY STEP — od zera do wyników na Snelliusie

Dla osoby, która ma konto na Snelliusie i nic poza tym. Wszystko leci na
partycji **rome** (CPU), żeby nie palić SBU. SMoRe ParS też puścimy na CPU
(rome) — jest lekki, GPU nie jest potrzebne.

Twój scratch: `/gpfs/scratch1/shared/jkowalczuk/`
Folder roboczy: `/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/combi3d/`

UWAGA: scratch kasuje pliki po 14 dniach. Po skończeniu sweepu skopiuj wyniki
gdzie indziej (home lub archive).

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ A — pobierz pliki na swój komputer
────────────────────────────────────────────────────────────────────────────

1. Pobierz `combi3D_pipeline.zip` (ten z czatu).
2. NIE rozpakowuj go u siebie. Wyślesz go na Snellius w całości.

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ B — zaloguj się na Snellius i wyślij pliki
────────────────────────────────────────────────────────────────────────────

3. Otwórz terminal (Mac: Terminal; Windows: PowerShell albo MobaXterm).

4. Wyślij zip na Snellius (zamień `jkowalczuk` na swój login jeśli inny):

   scp ~/Downloads/combi3D_pipeline.zip jkowalczuk@snellius.surf.nl:/gpfs/scratch1/shared/jkowalczuk/

   (wpisz hasło / przejdź 2FA gdy poprosi)

5. Zaloguj się na Snellius:

   ssh jkowalczuk@snellius.surf.nl

6. Przejdź na scratch i rozpakuj:

   cd /gpfs/scratch1/shared/jkowalczuk/
   mkdir -p surrogates/burns/combi3d
   mv combi3D_pipeline.zip surrogates/burns/combi3d/
   cd surrogates/burns/combi3d
   unzip combi3D_pipeline.zip
   cd combi3D
   ls          # powinnaś zobaczyć install_cc3d.slurm, test_run.slurm, verify.py, smore/ itd.

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ C — sprawdź że ścieżki w skryptach się zgadzają
────────────────────────────────────────────────────────────────────────────

7. Otwórz install_cc3d.slurm i sprawdź górę pliku:

   nano install_cc3d.slurm

   Zobacz linie:
       SCRATCH=/gpfs/scratch1/shared/jkowalczuk
       CONDA_ROOT=$SCRATCH/miniconda3
       ENV_NAME=cc3d
   Jeśli Twój login to faktycznie "jkowalczuk" — nic nie zmieniaj.
   Wyjdź: Ctrl+O, Enter, Ctrl+X.

   To samo sprawdź w test_run.slurm (te same zmienne + BASE).

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ D — zainstaluj wszystko (JEDEN RAZ, ~1-2h)
────────────────────────────────────────────────────────────────────────────

8. Wyślij job instalacyjny:

   sbatch install_cc3d.slurm

   Dostaniesz numer joba, np. "Submitted batch job 12345678".

9. Obserwuj postęp (Ctrl+C żeby przestać patrzeć, job leci dalej):

   tail -f install_12345678.out

   Czekasz aż zobaczysz "[install] DONE." i listę [ok] importów.
   Jeśli zobaczysz [FAIL] — skopiuj cały plik .out i pokaż go (Claude/promotor).

10. Sprawdź że job się skończył:

    squeue -u $USER        # jak pusto = skończony

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ E — jeden testowy run (sprawdza czy CC3D działa)
────────────────────────────────────────────────────────────────────────────

11. Wyślij test:

    sbatch test_run.slurm

12. Obserwuj:

    tail -f testrun_*.out

    Czekasz na "[test] SUCCESS -- CC3D works."
    Jeśli zobaczysz "[FAIL] no mean_concentration.txt" — pokaż plik .out.

    Ten krok robi też verify.py i generuje manifest.json + stażuje katalogi
    runów, więc po nim wszystko jest gotowe do pełnego sweepu.

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ F — pełny sweep (10 runów naraz)
────────────────────────────────────────────────────────────────────────────

13. test_run.slurm zapisał już skrypt array. Wyślij go:

    sbatch ../sweep/sweep_array.sh

    To odpala 10 runów równolegle (każdy ~1h). Sprawdzaj:

    squeue -u $USER         # zobaczysz 10 zadań run_0001..run_0010

14. Gdy wszystko zniknie z kolejki — sprawdź że są wyniki:

    ls ../sweep/outputs/run_0001/datafiles/mean_concentration.txt
    ls ../sweep/outputs/run_0010/datafiles/mean_concentration.txt

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ G — kalibracja (Sobol → SMoRe ParS), na CPU
────────────────────────────────────────────────────────────────────────────

15. To jest lekkie, leci w kilka minut. Możesz na nodzie interaktywnym:

    srun --partition=rome --cpus-per-task=8 --time=00:30:00 --pty bash
    source /gpfs/scratch1/shared/jkowalczuk/miniconda3/bin/activate cc3d
    cd /gpfs/scratch1/shared/jkowalczuk/surrogates/burns/combi3d/combi3D

    python smore/run_calibration.py --sim-root ../sweep/outputs \
        --manifest manifest.json --top-k 5 --out calibration_results.json

16. Wynik jest w calibration_results.json. Zobaczysz:
    - ranking Sobola (które parametry są czułe)
    - top-5 wybrane do kalibracji
    - recovery nRMSE / R2 per parametr

────────────────────────────────────────────────────────────────────────────
## CZĘŚĆ H — zachowaj wyniki (scratch kasuje po 14 dniach!)
────────────────────────────────────────────────────────────────────────────

17. Skopiuj wyniki gdzieś trwałego, np. do home:

    cp calibration_results.json ~/
    # albo cały sweep:
    tar czf ~/combi3d_sweep_$(date +%Y%m%d).tar.gz ../sweep/outputs calibration_results.json

────────────────────────────────────────────────────────────────────────────
## NAJCZĘSTSZE PROBLEMY
────────────────────────────────────────────────────────────────────────────

- "conda: command not found" w test_run → instalacja nie doszła do końca,
  sprawdź install_*.out.
- Job od razu kończy się błędem o partycji/koncie → dodaj swoje konto:
  w pliku .slurm dopisz na górze:  #SBATCH --account=TWOJE_KONTO
  (sprawdź konto: `accinfo` albo `sacct`)
- CC3D startuje ale brak mean_concentration.txt → najpewniej ścieżka do
  variablevals3D albo brak FiPy; pokaż output.
- "No space left" → scratch pełny, wyczyść stare pliki.

────────────────────────────────────────────────────────────────────────────
## SKĄD CO BIERZE PARAMETRY (gdyby promotor pytał)
────────────────────────────────────────────────────────────────────────────

setup_runs.py  → manifest.json (LHS, seed 42, deterministyczny)
run_sweep.py   → kopiuje kod do sweep/runs/run_XXXX/Simulation/ + params.json
CC3D           → czyta params.json przez param_loader (NIE env var, plik lokalny)
               → output do sweep/outputs/run_XXXX/ (poza katalogiem .cc3d)
smore/         → czyta outputs + params.json, robi Sobol potem SMoRe ParS

Manifest jest JEDYNYM źródłem prawdy. Żaden run nie może cicho pójść na
baseline — param_loader rzuca błąd jak czegoś brakuje.
