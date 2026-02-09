import os
import click
import time
import numpy as np
import sympy as sp
import pandas as pd
import yaml
import os

dt = 0.9
dim = 'dy'
default_csv = f'data/fixed/fixed_point_{int(dt*100)}_{dim}.csv'

@click.command()
@click.option("--experiment_name", default="fixed_pse", type=str, help="experiment_name")
@click.option("--yaml_file", default=f"=fixed_{int(dt*100)}_{dim}_result.yaml", type=str, help="experiment_name")
@click.option("--gpu_index", "-g", default=0, type=int, help="gpu index used")
# @click.option("--operators","-l",default="['Add','Mul','Sub','Div','Identity','Sin','Cos','Exp','Log']",help="operator library")
@click.option("--operators","-l",default="['Add','Mul','Sub','Identity']",help="operator library")  # Div
@click.option("--n_down_sample","-d",default=3000,type=int,help="n sample to downsample in PSRN for speeding up")
@click.option("--n_inputs","-i",default=5,type=int,help="PSRN input size (n variables + n constants)")
@click.option("--seed", "-s", default=0, type=int, help="seed")
@click.option("--topk","-k",default=10,type=int,help="number of best expressions to take from PSRN to fit")
@click.option("--use_constant", "-c", default=False, type=bool, help="use const in PSE")
@click.option("--probe","-o",default=None,type=str,help="expression probe, string, PSE will stop if probe is in pf")
@click.option("--csvpath","-q",default=default_csv,type=str,help="path to custom csv file")
@click.option("--use_cpu",default=False,type=bool,help="use cpu")
@click.option("--time_limit", default=3600, type=int, help="time limit (s)")
def main(experiment_name, yaml_file, gpu_index, operators, n_down_sample, n_inputs, seed, topk, use_constant, probe, csvpath, use_cpu, time_limit):
    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import torch
    from psrn import PSRN_Regressor

    if not use_cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    operators = eval(operators)
    df = pd.read_csv(csvpath, header=None)

    Input = df.values[:, :-1].reshape(len(df), -1)
    Output = df.values[:, -1].reshape(len(df), 1)

    variables_name = [f"x{i}" for i in range(Input.shape[1])]

    regressor = PSRN_Regressor(
        variables=variables_name,
        use_const=use_constant,
        device=device,
        token_generator_config={
            "base": {
                "has_const": use_constant,
                "tokens": operators
            }
        },
        stage_config={
            "default": {
                "operators": operators,
                "time_limit": time_limit,
                "n_psrn_inputs": n_inputs,
                "n_sample_variables": 3,
            },
            "stages": [
                {},
            ],
        },
    )

    start = time.time()
    flag, pareto_ls = regressor.fit(
        Input,
        Output,
        n_down_sample=n_down_sample,
        use_threshold=False,
        threshold=1e-20,
        probe=probe,
        prun_const=True,
        prun_ndigit=6,
        top_k=topk,
    )
    end = time.time()
    time_cost = end - start

    pareto_ls = regressor.display_expr_table(sort_by='mse') # or 'reward'

    expr_str, reward, loss, complexity = pareto_ls[0]
    print('Found:', expr_str, 'time_cost', time_cost)

    log_dir = os.path.join("log", "custom_data", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    result_config = {
        "experiment": experiment_name,
        "csv_path": csvpath,
        "time_cost_seconds": float(time_cost),
        "success": bool(flag),
        "best_expression": {
            "string_form": str(expr_str),
            "reward": float(reward),
            "loss": float(loss),
            "complexity": int(complexity),
        },
        "hyperparameters": {
            "n_inputs": n_inputs,
            "use_constant": use_constant,
            "operators": operators,
            "n_down_sample": n_down_sample,
            "topk": topk,
            "seed": seed,
        }
    }

    config_path = os.path.join(log_dir, yaml_file)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(result_config, f, allow_unicode=True, sort_keys=False)

    print(f"✅ Expression and config saved to: {config_path}")

if __name__ == "__main__":
    main()