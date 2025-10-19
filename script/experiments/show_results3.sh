cd ../../tools
seed=1
id=3
mixing="Mixing/results_year_${seed}_300s.csv"
mixsat="MIXSAT/results_year_${seed}_300s.csv"
# mixing="mixing_solver/results_year_2.csv"
# mixsat="mixsat_solver/results_year_2.csv"
gat="LS-GAT/results_${id}_year_${seed}_300s.csv"
sgat="LS-SGAT/results_11_year_${seed}_300s.csv"
# gat="LS-SGAT/results_10_year_${seed}.csv"
fouriersat="FourierSAT/results_year_${seed}_300s.csv"
# fouriersat="fourier_solver/results_year_1.csv"

python results_formatter.py \
    --files $mixing $mixsat $fouriersat $gat $sgat\
    --years 2020 2021 2022 2023 2024\
    --names Mixing MIXSAT FourierSAT LS-GAT LS-SGAT \
    --best_cost ../vba/vbayear_uw.csv ../solver_output ../solver_output/fig \
    --show_scores \
    --proposed_method 2 \
    --show-timeouts \
    --show-all \
    --add-unfound